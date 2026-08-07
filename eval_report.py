"""
Score every cached model x video pair against the labelled ground truth.

Inference is never re-run here -- eval_run_detect.py cached every box down to
cache_conf, so the confidence sweep is pure arithmetic.

    python eval_report.py --project projects/hph_hw
    python eval_report.py --project projects/hph_hw --conf 0.2,0.4,0.6
    python eval_report.py --project projects/hph_hw --no-chart

Metrics
  frame level   multiset precision / recall / F1 per class, plus a confusion matrix
  segment level temporal-IoU-matched F1 at several thresholds, plus an edit score
  timing        onset latency to the first stable detection, and flicker rate
  counting      MAE against the instance counts entered while labelling
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from eval_common import (
    NONE_CLASS,
    get_roi,
    gt_frame_counts,
    gt_segments_for,
    levenshtein,
    load_classes,
    load_det_cache,
    load_project,
    pred_frame_counts,
    qc_chart_lanes,
    qc_lane_names,
    rle_segments,
    sanitize,
    smooth_segments,
    temporal_iou,
)


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


# --------------------------------------------------------------------------
# Frame level
# --------------------------------------------------------------------------

def frame_metrics(G, P, classes):
    """
    Per-class precision / recall / F1 treating each frame as a MULTISET of
    instances rather than a presence flag.

        TP = sum min(G, P)   FP = sum max(0, P-G)   FN = sum max(0, G-P)

    With every count equal to 1 this collapses to the usual binary frame-level
    metric, so it is a strict generalisation -- but it also catches a model that
    reports one screen where the operator labelled two.
    """
    common = np.minimum(G, P)
    tp = common.sum(axis=0).astype(np.int64)
    fp = np.maximum(0, P - G).sum(axis=0).astype(np.int64)
    fn = np.maximum(0, G - P).sum(axis=0).astype(np.int64)

    out = []
    for k in range(G.shape[1]):
        precision = _safe_div(tp[k], tp[k] + fp[k])
        recall = _safe_div(tp[k], tp[k] + fn[k])
        out.append({
            "class": classes[k],
            "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]),
            "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "support_frames": int((G[:, k] > 0).sum()),
            "gt_frames": int((G[:, k] > 0).sum()),
            "pred_frames": int((P[:, k] > 0).sum()),
        })
    return out


def confusion_matrix(G, P, classes):
    """
    (K+1)x(K+1) confusion over classes plus __none__.

    Overlapping multi-label ground truth admits no unique frame -> single-class
    assignment, so only the diagonal and the __none__ row/column are exact. The
    off-diagonal cells come from a deterministic greedy pairing of the leftover
    GT and predicted instances (largest first, ties by class index) and should be
    read as a hint about which classes get confused, not as a hard count.
    """
    K = len(classes)
    M = np.zeros((K + 1, K + 1), dtype=np.int64)
    common = np.minimum(G, P)
    np.fill_diagonal(M[:K, :K], common.sum(axis=0))

    left_g = G - common
    left_p = P - common
    rows = np.flatnonzero(left_g.any(axis=1) | left_p.any(axis=1))

    for t in rows:
        u = left_g[t].copy()
        v = left_p[t].copy()
        while u.any() and v.any():
            gi = int(np.argmax(u))
            pi = int(np.argmax(v))
            M[gi, pi] += 1
            u[gi] -= 1
            v[pi] -= 1
        for gi in np.flatnonzero(u):
            M[gi, K] += int(u[gi])          # missed
        for pi in np.flatnonzero(v):
            M[K, pi] += int(v[pi])          # false alarm

    labels = list(classes) + [NONE_CLASS]
    return pd.DataFrame(M, index=labels, columns=labels)


# --------------------------------------------------------------------------
# Segment level
# --------------------------------------------------------------------------

def match_segments(pred_segs, gt_segs, thr):
    """Greedy one-to-one matching by descending temporal IoU. Same class only."""
    pairs = []
    for pi, ps in enumerate(pred_segs):
        for gi, gs in enumerate(gt_segs):
            iou = temporal_iou(ps, gs)
            if iou >= thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    used_p, used_g, matched = set(), set(), []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((pi, gi, iou))

    return len(matched), len(pred_segs) - len(matched), len(gt_segs) - len(matched), matched


def edit_score_tokens(pred_by_class, gt_by_class, classes):
    """
    Levenshtein-based edit score adapted to overlapping multi-label timelines.

    The textbook action-segmentation edit score collapses a mutually-exclusive
    per-frame label sequence, which does not exist here. Reducing each timeline to
    per-class binary sequences does not work either: collapsing a binary presence
    signal always yields strictly alternating runs, so the distance degenerates to
    a difference in run counts -- flicker rate already measures that, and it is
    blind to when anything happened.

    Instead both timelines are reduced to the sequence of class tokens ordered by
    segment start frame. That is well defined even when segments overlap, and it
    preserves the question the metric was invented to ask: did the model see the
    right sequence of workflow steps, ignoring how long each one took.
    """
    def tokens(by_class):
        items = []
        for k, cls in enumerate(classes):
            for start, end in by_class.get(cls, []):
                items.append((start, k, cls))
        return [cls for _, _, cls in sorted(items)]

    pred_tokens = tokens(pred_by_class)
    gt_tokens = tokens(gt_by_class)
    denom = max(len(pred_tokens), len(gt_tokens))
    if denom == 0:
        return 100.0
    return (1.0 - levenshtein(pred_tokens, gt_tokens) / denom) * 100.0


def onset_latency(present, gt_segs, window):
    """
    Frames from each GT segment's start to the first stable detection, where
    "stable" means `window` consecutive frames of the class being predicted.

    A single-frame blip therefore does not count as detection. Segments shorter
    than the window fall back to a single frame so they are still scored.
    Returns a list with NaN for segments that were never stably detected.
    """
    latencies = []
    for start, end, _count in gt_segs:
        span = end - start + 1
        w = min(window, span) if span < window else window
        hit = np.nan
        for t in range(start, end + 1):
            stop = min(end + 1, t + w)
            if stop - t < w:
                break
            if present[t:stop].all():
                hit = t - start
                break
        latencies.append(float(hit))
    return latencies


def flicker_stats(raw_segs, gt_segs):
    """Raw (unsmoothed) predicted runs overlapping each GT segment. 1.0 is perfect."""
    if not gt_segs:
        return float("nan"), _safe_div(len(raw_segs), 1) if raw_segs else 0.0
    fragments = [
        sum(1 for ps in raw_segs if ps[1] >= gs[0] and ps[0] <= gs[1])
        for gs in gt_segs
    ]
    return float(np.mean(fragments)), _safe_div(len(raw_segs), len(gt_segs))


def count_metrics(G_k, P_k, gt_segs):
    active = G_k > 0
    diff = (P_k.astype(np.int64) - G_k.astype(np.int64))
    seg_errors = []
    for start, end, count in gt_segs:
        window = P_k[start:end + 1]
        if window.size == 0:
            continue
        mode = int(np.bincount(window.astype(np.int64)).argmax())
        seg_errors.append(abs(mode - count))
    return {
        "count_mae_active": float(np.abs(diff[active]).mean()) if active.any() else 0.0,
        "count_mae_all": float(np.abs(diff).mean()) if diff.size else 0.0,
        "count_bias": float(diff[active].mean()) if active.any() else 0.0,
        "count_exact_acc": float((P_k[active] == G_k[active]).mean()) if active.any() else 0.0,
        "seg_count_mae": float(np.mean(seg_errors)) if seg_errors else 0.0,
    }


# --------------------------------------------------------------------------
# Evaluation of one (model, video, roi, conf)
# --------------------------------------------------------------------------

def evaluate_pair(cfg, classes, gt, df, meta, roi_name, conf_thr):
    """Score one combination. Returns frame rows, segment rows, summary, confusion, chart data."""
    roi = get_roi(gt, roi_name)
    total_frames = int(gt["total_frames"])
    fps = float(gt["fps"]) or 1.0

    G, gt_warnings = gt_frame_counts(gt, roi_name, classes)
    P, C, stats = pred_frame_counts(
        df, meta, roi, classes, conf_thr, total_frames, cfg.get("class_alias", {})
    )

    frame_rows = frame_metrics(G, P, classes)
    thresholds = cfg["seg_iou_thresholds"]
    merge_gap = cfg["merge_gap_frames"]
    min_len = cfg["min_segment_frames"]
    window = cfg["stability_window"]

    seg_rows = []
    pred_by_class, gt_by_class, chart_segments = {}, {}, {}

    for k, cls in enumerate(classes):
        raw_segs = rle_segments(P[:, k] > 0)
        pred_segs = smooth_segments(raw_segs, merge_gap, min_len)
        gt_segs = gt_segments_for(gt, roi_name, cls)
        pred_by_class[cls] = pred_segs
        gt_by_class[cls] = [(s, e) for s, e, _ in gt_segs]

        row = {"class": cls, "n_gt_seg": len(gt_segs),
               "n_pred_seg": len(pred_segs), "n_pred_seg_raw": len(raw_segs)}
        for thr in thresholds:
            tp, fp, fn, _ = match_segments(pred_segs, [(s, e) for s, e, _ in gt_segs], thr)
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            row[f"f1_iou{int(thr * 100):02d}"] = _safe_div(
                2 * precision * recall, precision + recall
            )

        latencies = onset_latency(P[:, k] > 0, gt_segs, window)
        found = [x for x in latencies if not np.isnan(x)]
        row["onset_median_frames"] = float(np.median(found)) if found else float("nan")
        row["onset_p90_frames"] = float(np.percentile(found, 90)) if found else float("nan")
        row["onset_median_sec"] = row["onset_median_frames"] / fps
        row["onset_miss_rate"] = _safe_div(len(latencies) - len(found), len(latencies))

        flicker_mean, over_seg = flicker_stats(raw_segs, [(s, e) for s, e, _ in gt_segs])
        row["flicker_mean"] = flicker_mean
        row["over_seg_ratio"] = over_seg

        # Per-frame counting errors belong with the frame metrics; the per-segment
        # count error is the segment-level answer to the same question.
        counts = count_metrics(G[:, k], P[:, k], gt_segs)
        row["seg_count_mae"] = counts.pop("seg_count_mae")
        frame_rows[k].update(counts)
        seg_rows.append(row)

        chart_segments[cls] = [
            (s, e, float(C[s:e + 1, k][C[s:e + 1, k] > 0].mean())
             if (C[s:e + 1, k] > 0).any() else 0.0)
            for s, e in pred_segs
        ]

    # Chart-only: the QC lanes carry no metrics, because scoring them needs an NG
    # ground-truth schema the labeller does not have yet. They ride on the same
    # detections as the class lane above them, just split by verdict.
    chart_segments.update(
        qc_chart_lanes(cfg, classes, df, meta, roi, conf_thr, total_frames,
                       cfg.get("class_alias", {}))
    )

    supported = [i for i, r in enumerate(frame_rows) if r["support_frames"] > 0]
    macro_f1 = float(np.mean([frame_rows[i]["f1"] for i in supported])) if supported else 0.0
    tp_all = sum(r["tp"] for r in frame_rows)
    fp_all = sum(r["fp"] for r in frame_rows)
    fn_all = sum(r["fn"] for r in frame_rows)
    micro_p = _safe_div(tp_all, tp_all + fp_all)
    micro_r = _safe_div(tp_all, tp_all + fn_all)

    seg_supported = [r for r in seg_rows if r["n_gt_seg"] > 0]
    onsets = [r["onset_median_frames"] for r in seg_supported
              if not np.isnan(r["onset_median_frames"])]
    flickers = [r["flicker_mean"] for r in seg_supported if not np.isnan(r["flicker_mean"])]

    def macro(field):
        return float(np.mean([r[field] for r in seg_supported])) if seg_supported else 0.0

    summary = {
        "frame_macro_f1": macro_f1,
        "frame_micro_f1": _safe_div(2 * micro_p * micro_r, micro_p + micro_r),
        "frame_micro_precision": micro_p,
        "frame_micro_recall": micro_r,
        "edit_score": edit_score_tokens(pred_by_class, gt_by_class, classes),
        "onset_median_frames": float(np.median(onsets)) if onsets else float("nan"),
        "onset_median_sec": (float(np.median(onsets)) / fps) if onsets else float("nan"),
        "flicker_mean": float(np.mean(flickers)) if flickers else float("nan"),
        "count_mae_active": float(
            np.mean([frame_rows[i]["count_mae_active"] for i in supported])
        ) if supported else 0.0,
        "seg_count_mae": macro("seg_count_mae"),
        "n_gt_segments": sum(r["n_gt_seg"] for r in seg_rows),
    }
    for thr in thresholds:
        key = f"f1_iou{int(thr * 100):02d}"
        summary[f"seg_{key}"] = macro(key)

    weights = cfg.get("score_weights", {})
    summary["score"] = (
        weights.get("frame_macro_f1", 0.5) * summary["frame_macro_f1"]
        + weights.get("seg_f1_025", 0.5) * summary.get("seg_f1_iou25", 0.0)
    )

    return {
        "frame_rows": frame_rows,
        "seg_rows": seg_rows,
        "summary": summary,
        "confusion": confusion_matrix(G, P, classes),
        "chart_segments": chart_segments,
        "stats": stats,
        "gt_warnings": gt_warnings,
    }


# --------------------------------------------------------------------------
# Report driver
# --------------------------------------------------------------------------

def write_leaderboard(path, summary_df, cfg, classes):
    lines = [
        f"# Evaluation leaderboard -- {cfg['project']}",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}  |  "
        f"{len(classes)} classes  |  score = "
        f"{cfg['score_weights'].get('frame_macro_f1', 0.5)}*frame_macro_f1 + "
        f"{cfg['score_weights'].get('seg_f1_025', 0.5)}*seg_f1@0.25",
        "",
        "## Best confidence per model",
        "",
        "| model | best conf | score | frame macro F1 | seg F1@0.25 | edit | onset (s) | flicker |",
        "|---|---|---|---|---|---|---|---|",
    ]

    by_model = summary_df.groupby(["model", "conf"], as_index=False).mean(numeric_only=True)
    best = by_model.loc[by_model.groupby("model")["score"].idxmax()].sort_values(
        "score", ascending=False
    )
    for _, r in best.iterrows():
        lines.append(
            f"| {r['model']} | {r['conf']:.2f} | **{r['score']:.3f}** | "
            f"{r['frame_macro_f1']:.3f} | {r['seg_f1_iou25']:.3f} | {r['edit_score']:.1f} | "
            f"{r['onset_median_sec']:.2f} | {r['flicker_mean']:.2f} |"
        )

    lines += ["", "## Full sweep", "",
              "| model | video | roi | conf | score | frame macro F1 | seg F1@0.25 | edit |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in summary_df.sort_values("score", ascending=False).iterrows():
        lines.append(
            f"| {r['model']} | {r['video']} | {r['roi']} | {r['conf']:.2f} | {r['score']:.3f} | "
            f"{r['frame_macro_f1']:.3f} | {r['seg_f1_iou25']:.3f} | {r['edit_score']:.1f} |"
        )

    lines += [
        "",
        "> Confusion-matrix off-diagonal cells use a greedy pairing of leftover instances; "
        "the diagonal and the `__none__` row/column are exact.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return best


def main():
    parser = argparse.ArgumentParser(description="Score cached detections against ground truth.")
    parser.add_argument("--project", required=True, help="Path to projects/<name>")
    parser.add_argument("--conf", help="Comma-separated thresholds (default: config conf_sweep)")
    parser.add_argument("--models", help="Comma-separated model names (default: all)")
    parser.add_argument("--videos", help="Comma-separated video names (default: all)")
    parser.add_argument("--gantt-conf", type=float, help="Force the chart confidence threshold")
    parser.add_argument("--gantt-all-conf", action="store_true",
                        help="Render a chart for every threshold in the sweep")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--dim-mode", choices=["blend", "alpha"], default="blend",
                        help="How confidence dims the chart colours (default blend)")
    args = parser.parse_args()

    cfg = load_project(args.project)
    classes = load_classes(args.project, cfg)
    sweep = ([float(c) for c in args.conf.split(",")] if args.conf else
             [float(c) for c in cfg["conf_sweep"]])
    below = [c for c in sweep if c < float(cfg["cache_conf"])]
    if below:
        print(f"❌ thresholds {below} are below cache_conf={cfg['cache_conf']}; "
              f"those boxes are not in the cache.")
        return

    models = cfg["models"]
    videos = cfg["videos"]
    if args.models:
        wanted = {s.strip() for s in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]
    if args.videos:
        wanted = {s.strip() for s in args.videos.split(",")}
        videos = [v for v in videos if v["name"] in wanted]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.project, "reports", run_id)
    os.makedirs(out_dir, exist_ok=True)

    frame_records, seg_records, summary_records = [], [], []
    chart_data = {}
    mapping_notes = {}
    skipped = []

    for v in videos:
        try:
            gt = load_gt_or_skip(args.project, v["name"])
        except FileNotFoundError as exc:
            skipped.append(str(exc))
            continue
        if not gt["rois"]:
            skipped.append(f"{v['name']}: ground truth has no ROIs")
            continue

        for m in models:
            try:
                df, meta = load_det_cache(args.project, m["name"], v["name"])
            except FileNotFoundError as exc:
                skipped.append(str(exc))
                continue

            if abs(float(meta["fps"]) - float(gt["fps"])) > 0.01:
                print(f"❌ {m['name']} x {v['name']}: fps mismatch -- ground truth {gt['fps']} vs "
                      f"cache {meta['fps']}. The ground truth was labelled against a different "
                      f"encode; re-label or re-cache.")
                continue
            if meta["frames_decoded"] != gt["total_frames"]:
                print(f"   ⚠ {m['name']} x {v['name']}: cache decoded {meta['frames_decoded']} "
                      f"frames, ground truth spans {gt['total_frames']}.")

            for roi in gt["rois"]:
                for conf in sweep:
                    res = evaluate_pair(cfg, classes, gt, df, meta, roi["name"], conf)
                    key = dict(model=m["name"], video=v["name"], roi=roi["name"], conf=conf)

                    for row in res["frame_rows"]:
                        frame_records.append({**key, **row})
                    for row in res["seg_rows"]:
                        seg_records.append({**key, **row})
                    summary_records.append({**key, **res["summary"]})
                    chart_data[(v["name"], roi["name"], conf, m["name"])] = res["chart_segments"]

                    tag = f"{sanitize(m['name'])}_{sanitize(v['name'])}_{sanitize(roi['name'])}"
                    res["confusion"].to_csv(
                        os.path.join(out_dir, f"confusion_{tag}_conf{conf:.2f}.csv")
                    )
                    mapping_notes[f"{m['name']}__{v['name']}"] = res["stats"]
                    for warn in res["gt_warnings"]:
                        print(f"   ⚠ {v['name']}/{roi['name']}: {warn}")

    if not summary_records:
        print("❌ Nothing scored. Check that both the detection cache and the ground truth exist.")
        for note in skipped:
            print(f"   {note}")
        return

    frame_df = pd.DataFrame(frame_records)
    seg_df = pd.DataFrame(seg_records)
    summary_df = pd.DataFrame(summary_records)
    frame_df.to_csv(os.path.join(out_dir, "metrics_frame.csv"), index=False)
    seg_df.to_csv(os.path.join(out_dir, "metrics_segment.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    best = write_leaderboard(os.path.join(out_dir, "leaderboard.md"), summary_df, cfg, classes)

    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "project": cfg["project"],
            "classes": classes,
            "conf_sweep": sweep,
            "config": {k: val for k, val in cfg.items() if not k.startswith("_")},
            "class_mapping": {k: {
                "unmapped_model_classes": s["unmapped_model_classes"],
                "project_classes_not_in_model": s["project_classes_not_in_model"],
                "rows_total": s["rows_total"],
                "rows_kept": s["rows_kept"],
                "rows_dropped_unmapped": s["rows_dropped_unmapped"],
            } for k, s in mapping_notes.items()},
            "skipped": skipped,
            "confusion_note": (
                "Off-diagonal cells use a greedy pairing of leftover instances; the diagonal "
                "and the __none__ row/column are exact."
            ),
        }, f, indent=2)

    print(f"\n📊 Leaderboard  ({len(summary_df)} combinations scored)")
    for _, r in best.iterrows():
        print(f"   {r['model']:<24} conf={r['conf']:.2f}  score={r['score']:.3f}  "
              f"frameF1={r['frame_macro_f1']:.3f}  segF1@.25={r['seg_f1_iou25']:.3f}  "
              f"edit={r['edit_score']:.1f}  onset={r['onset_median_sec']:.2f}s  "
              f"flicker={r['flicker_mean']:.2f}")
    for note in skipped:
        print(f"   ⚠ skipped: {note}")

    if not args.no_chart:
        render_charts(args, cfg, classes, summary_df, chart_data, out_dir, sweep)

    print(f"\n✅ Report written to {out_dir}")


def load_gt_or_skip(project_dir, video_name):
    from eval_common import load_gt

    gt = load_gt(project_dir, video_name)
    if gt is None:
        raise FileNotFoundError(f"{video_name}: no ground truth -- run eval_label_timeline.py")
    return gt


def render_charts(args, cfg, classes, summary_df, chart_data, out_dir, sweep):
    """Render one Gantt per (video, roi) at the most informative threshold."""
    try:
        from generate_eval_gantt import build_gantt, write_figure
    except ImportError as exc:
        print(f"⚠ chart skipped, plotly unavailable: {exc}")
        return

    from eval_common import load_gt

    for (video, roi), group in summary_df.groupby(["video", "roi"]):
        if args.gantt_all_conf:
            confs = sweep
        elif args.gantt_conf is not None:
            confs = [args.gantt_conf]
        else:
            # the threshold at which the models collectively look their best
            confs = [float(group.groupby("conf")["score"].mean().idxmax())]

        gt = load_gt(args.project, video)
        for conf in confs:
            per_model = {
                model: chart_data[(video, roi, conf, model)]
                for (v, r, c, model) in chart_data
                if v == video and r == roi and abs(c - conf) < 1e-9
            }
            if not per_model:
                continue
            fig = build_gantt(
                gt=gt, roi_name=roi, per_model_segments=per_model,
                classes=classes + qc_lane_names(cfg, classes),
                conf_thr=conf, title=f"{cfg['project']} -- {video} / {roi} @ conf {conf:.2f}",
                dim_mode=args.dim_mode,
            )
            stem = os.path.join(
                out_dir, f"gantt_{sanitize(video)}_{sanitize(roi)}_conf{conf:.2f}"
            )
            write_figure(fig, stem)


if __name__ == "__main__":
    main()
