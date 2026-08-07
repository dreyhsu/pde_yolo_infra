"""
Calibrate the screen_qc thresholds on real footage, and self-test them on the three
reference screenshots.

The defaults in screen_qc.DEFAULT_QC were fitted to exactly three frames. They have wide
margins, but three frames is not evidence -- before trusting them on the line, dump the
features over real clips and look at where the distributions actually sit.

    # 1. sanity check, no weights or video needed (runs anywhere cv2 is installed)
    python calibrate_screen_qc.py --selftest

    # 2. one pass per clip you can label as a whole
    python calibrate_screen_qc.py --weights runs/best.pt --source ok_clip.mp4 \
        --label ok --out calib/
    python calibrate_screen_qc.py --weights runs/best.pt --source bare_clip.mp4 \
        --label no_plastic --out calib/
    python calibrate_screen_qc.py --weights runs/best.pt --source flip_clip.mp4 \
        --label flipped --out calib/

    # 3. pick thresholds across every labelled dump
    python calibrate_screen_qc.py --suggest calib/*.qc.csv

Each dump also saves the core crops it measured plus a contact sheet, so any surprising
row can be eyeballed instead of guessed at.
"""

import argparse
import csv
import glob
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

from screen_qc import (
    DEFAULT_QC,
    FEATURE_COLUMNS,
    QC_FLIPPED,
    QC_NO_PLASTIC,
    QC_OK,
    QC_UNKNOWN,
    QC_VERDICTS,
    _clamp_box,
    _inset,
    classify_features,
    merge_qc_config,
    screeninbox_features,
)

CSV_COLUMNS = [
    "source", "frame", "conf", "x1", "y1", "x2", "y2",
    *FEATURE_COLUMNS, "core_median", "ring_median", "label", "verdict",
]

LABELS = (QC_OK, QC_NO_PLASTIC, QC_FLIPPED)


# --------------------------------------------------------------------------
# Self-test on the annotated reference screenshots
# --------------------------------------------------------------------------

REFERENCE_DIR = os.path.join("transfer", "screen_ng")
REFERENCE_EXPECTED = {
    "normal": QC_OK,
    "plastic_ng": QC_NO_PLASTIC,
    "flip_ng": QC_FLIPPED,
}


def annotation_bbox(image_bgr):
    """
    Recover the box Ultralytics drew on a saved prediction screenshot.

    The reference PNGs are annotated screenshots, not raw frames -- there is no label
    file to read, so the box is recovered from the annotation colour itself. The class
    banner bleeds off the right edge of every screenshot, so the horizontal extent is
    taken from a row near the BOTTOM of the box (below the banner) rather than from the
    full colour mask.
    """
    b, g, r = (image_bgr[:, :, i].astype(np.int16) for i in range(3))
    mask = (b > 90) & (b - r > 50) & (b - g > 50)
    if not mask.any():
        return None
    y2 = int(np.nonzero(mask)[0].max())
    cols = np.nonzero(mask[max(0, y2 - 2)])[0]
    if cols.size == 0:
        return None
    x1, x2 = int(cols.min()), int(cols.max())
    rows = np.nonzero(mask[:, min(x1 + 2, mask.shape[1] - 1)])[0]
    if rows.size == 0:
        return None
    return x1, int(rows.min()), x2, y2


def run_selftest(cfg, ref_dir=REFERENCE_DIR):
    print(f"self-test on {ref_dir}\n")
    header = f"{'image':<12} {'verdict':<12} {'expected':<12} " + " ".join(
        f"{c:>11}" for c in FEATURE_COLUMNS
    )
    print(header)
    print("-" * len(header))

    failures = []
    for stem, expected in REFERENCE_EXPECTED.items():
        path = os.path.join(ref_dir, f"{stem}.png")
        image = cv2.imread(path)
        if image is None:
            failures.append(f"{stem}: could not read {path}")
            continue
        box = annotation_bbox(image)
        if box is None:
            failures.append(f"{stem}: no annotation box found in {path}")
            continue
        feats = screeninbox_features(image, box, cfg)
        verdict = classify_features(feats, cfg)
        vals = " ".join(f"{feats[c]:>11.3f}" for c in FEATURE_COLUMNS) if feats else ""
        flag = " " if verdict == expected else "  <-- MISMATCH"
        print(f"{stem:<12} {verdict:<12} {expected:<12} {vals}{flag}")
        if verdict != expected:
            failures.append(f"{stem}: got {verdict}, expected {expected}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("PASS  all three reference images classified correctly")
    return 0


# --------------------------------------------------------------------------
# Feature dump
# --------------------------------------------------------------------------

def screeninbox_ids(model_names, class_name):
    """Model class ids whose name matches `class_name` (case/separator insensitive)."""
    def norm(s):
        return str(s).strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    target = norm(class_name)
    return {int(k) for k, v in model_names.items() if norm(v) == target}


def core_crop(image_bgr, box, cfg):
    """The exact region the features were measured on, for visual review."""
    clamped = _clamp_box(box, image_bgr.shape[1], image_bgr.shape[0])
    if clamped is None:
        return None
    x1, y1, x2, y2 = _inset(clamped, cfg["core_inset"])
    crop = image_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def contact_sheet(crops, cell=(160, 100), cols=8):
    """Tile crops into one reviewable image."""
    if not crops:
        return None
    rows = (len(crops) + cols - 1) // cols
    sheet = np.zeros((rows * cell[1], cols * cell[0], 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        sheet[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = cv2.resize(
            crop, cell, interpolation=cv2.INTER_AREA
        )
    return sheet


def dump_features(weights, source, label, out_dir, cfg, class_name="screeninbox",
                  conf=0.25, imgsz=640, device="cuda", save_crops=True, max_crops=200):
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    csv_path = os.path.join(out_dir, f"{stem}__{label}.qc.csv")
    crop_dir = os.path.join(out_dir, f"{stem}__{label}.crops")
    if save_crops:
        os.makedirs(crop_dir, exist_ok=True)

    model = YOLO(weights)
    names = {str(k): v for k, v in model.names.items()}
    want = screeninbox_ids(names, class_name)
    if not want:
        raise SystemExit(
            f"model has no class named {class_name!r}; it has {sorted(names.values())}"
        )

    counts = {v: 0 for v in QC_VERDICTS}
    crops = []
    n_rows = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)

        # stream=True for the same reason as eval_run_detect.py: a non-streaming
        # predict() keeps every Results object (and its image tensor) alive at once.
        results = model.predict(
            source=source, conf=conf, imgsz=imgsz, device=device,
            save=False, show=False, stream=True, verbose=False,
        )
        for frame_idx, r in enumerate(results):
            boxes = r.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                cnf = boxes.conf.cpu().numpy()
                for box, c, p in zip(xyxy, cls, cnf):
                    if int(c) not in want:
                        continue
                    feats = screeninbox_features(r.orig_img, box, cfg)
                    verdict = classify_features(feats, cfg)
                    counts[verdict] += 1
                    writer.writerow([
                        stem, frame_idx, round(float(p), 4),
                        *(int(round(v)) for v in box),
                        *([round(feats[k], 6) for k in FEATURE_COLUMNS] if feats
                          else [""] * len(FEATURE_COLUMNS)),
                        round(feats["core_median"], 2) if feats else "",
                        round(feats["ring_median"], 2) if feats else "",
                        label, verdict,
                    ])
                    n_rows += 1
                    if save_crops and len(crops) < max_crops:
                        crop = core_crop(r.orig_img, box, cfg)
                        if crop is not None:
                            crops.append(crop)
                            cv2.imwrite(
                                os.path.join(crop_dir, f"{frame_idx:06d}_{verdict}.jpg"),
                                crop,
                            )
            del r

    print(f"💾 {n_rows} {class_name} detections -> {csv_path}")
    for v in QC_VERDICTS:
        if counts[v]:
            share = 100.0 * counts[v] / max(n_rows, 1)
            mark = "  <-- expected" if v == label else ""
            print(f"   {v:<12} {counts[v]:6d}  ({share:5.1f}%){mark}")
    if counts[QC_UNKNOWN]:
        print(f"   ⚠ {counts[QC_UNKNOWN]} detections were unmeasurable (box too small, "
              f"clipped, or unlit tray) -- they are excluded from --suggest")

    sheet = contact_sheet(crops)
    if sheet is not None:
        sheet_path = os.path.join(out_dir, f"{stem}__{label}.sheet.jpg")
        cv2.imwrite(sheet_path, sheet)
        print(f"   🖼 {len(crops)} crops + contact sheet -> {sheet_path}")

    return csv_path


# --------------------------------------------------------------------------
# Threshold suggestion
# --------------------------------------------------------------------------

def read_dumps(paths):
    """Labelled, measurable rows from every dump CSV."""
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("dark_ratio") or not row.get("mark_frac"):
                    continue  # unmeasurable -- has no opinion to score
                if row.get("label") not in LABELS:
                    continue
                rows.append({
                    "label": row["label"],
                    **{c: float(row[c]) for c in FEATURE_COLUMNS if row.get(c)},
                })
    return rows


def describe(rows, feature):
    print(f"\n  {feature}")
    print(f"    {'label':<12} {'n':>6} {'min':>9} {'p5':>9} {'median':>9} "
          f"{'p95':>9} {'max':>9}")
    for label in LABELS:
        vals = np.array([r[feature] for r in rows if r["label"] == label and feature in r])
        if vals.size == 0:
            continue
        print(f"    {label:<12} {vals.size:>6} {vals.min():>9.3f} "
              f"{np.percentile(vals, 5):>9.3f} {np.median(vals):>9.3f} "
              f"{np.percentile(vals, 95):>9.3f} {vals.max():>9.3f}")


def best_split(positives, negatives, direction):
    """
    Threshold maximising balanced accuracy of a 1-D split, plus the margin between the
    two distributions. `direction` is "below" when positives sit below the threshold.

    Balanced accuracy, not raw accuracy: defect clips are far shorter than OK clips, and
    a raw-accuracy optimum would happily call every frame OK.
    """
    if positives.size == 0 or negatives.size == 0:
        return None
    candidates = np.unique(np.concatenate([positives, negatives]))
    mids = (candidates[:-1] + candidates[1:]) / 2.0 if candidates.size > 1 else candidates
    best = None
    for t in mids:
        if direction == "below":
            tpr = float((positives < t).mean())
            tnr = float((negatives >= t).mean())
        else:
            tpr = float((positives > t).mean())
            tnr = float((negatives <= t).mean())
        score = (tpr + tnr) / 2.0
        if best is None or score > best[1]:
            best = (float(t), score, tpr, tnr)
    if direction == "below":
        gap = (float(negatives.min()) - float(positives.max()), positives.max(), negatives.min())
    else:
        gap = (float(positives.min()) - float(negatives.max()), negatives.max(), positives.min())
    return best, gap


def suggest(paths, cfg):
    rows = read_dumps(paths)
    if not rows:
        print("❌ no labelled, measurable rows found in the given CSVs")
        return 1

    n_by_label = {l: sum(1 for r in rows if r["label"] == l) for l in LABELS}
    print(f"{len(rows)} labelled rows: " + ", ".join(f"{l}={n_by_label[l]}" for l in LABELS))
    for feature in FEATURE_COLUMNS:
        describe(rows, feature)

    print("\nsuggested thresholds")

    # dark_ratio splits no_plastic (low) from everything else.
    pos = np.array([r["dark_ratio"] for r in rows if r["label"] == QC_NO_PLASTIC])
    neg = np.array([r["dark_ratio"] for r in rows if r["label"] != QC_NO_PLASTIC])
    dark_t = cfg["dark_ratio_max"]
    res = best_split(pos, neg, "below")
    if res:
        (t, score, tpr, tnr), (gap, hi_pos, lo_neg) = res
        dark_t = t
        print(f"  dark_ratio_max = {t:.3f}   (was {cfg['dark_ratio_max']}) "
              f"bal.acc={score:.3f}  recall={tpr:.3f}  spec={tnr:.3f}")
        print(f"    no_plastic max {hi_pos:.3f} | others min {lo_neg:.3f} -> "
              f"gap {gap:+.3f}" + ("  ⚠ distributions overlap" if gap <= 0 else ""))
    else:
        print(f"  dark_ratio_max: need both no_plastic and non-no_plastic rows")

    # mark_frac splits flipped from ok, but ONLY among rows that survive the darkness
    # test -- that is the order classify_features() applies, so calibrate it that way.
    survivors = [r for r in rows if r["dark_ratio"] >= dark_t]
    pos = np.array([r["mark_frac"] for r in survivors if r["label"] == QC_FLIPPED])
    neg = np.array([r["mark_frac"] for r in survivors if r["label"] == QC_OK])
    flip_t = cfg["flip_marks_min"]
    res = best_split(pos, neg, "above")
    if res:
        (t, score, tpr, tnr), (gap, hi_neg, lo_pos) = res
        flip_t = t
        print(f"  flip_marks_min = {t:.3f}   (was {cfg['flip_marks_min']}) "
              f"bal.acc={score:.3f}  recall={tpr:.3f}  spec={tnr:.3f}")
        print(f"    ok max {hi_neg:.3f} | flipped min {lo_pos:.3f} -> "
              f"gap {gap:+.3f}" + ("  ⚠ distributions overlap" if gap <= 0 else ""))
    else:
        print(f"  flip_marks_min: need both flipped and ok rows past the darkness test")

    tuned = merge_qc_config({"dark_ratio_max": dark_t, "flip_marks_min": flip_t})
    print("\nconfusion matrix with the suggested thresholds (rows = true label)")
    width = max(len(v) for v in QC_VERDICTS) + 2
    print("  " + " " * 12 + "".join(f"{v:>{width}}" for v in QC_VERDICTS))
    for label in LABELS:
        line = f"  {label:<12}"
        subset = [r for r in rows if r["label"] == label]
        for v in QC_VERDICTS:
            line += f"{sum(1 for r in subset if classify_features(r, tuned) == v):>{width}}"
        print(line)

    print(f"\nUpdate screen_qc.DEFAULT_QC (or the project's screen_qc config block) with:")
    print(f'  "dark_ratio_max": {dark_t:.3f},')
    print(f'  "flip_marks_min": {flip_t:.3f},')
    return 0


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="classify transfer/screen_ng/*.png and check the verdicts")
    p.add_argument("--suggest", nargs="+", metavar="CSV",
                   help="pick thresholds from one or more labelled dump CSVs")
    p.add_argument("--weights", help="YOLO .pt for a feature dump")
    p.add_argument("--source", help="video or image folder for a feature dump")
    p.add_argument("--label", choices=LABELS,
                   help="ground-truth label for every screeninbox in --source")
    p.add_argument("--out", default="calib", help="output directory (default: calib)")
    p.add_argument("--class-name", default="screeninbox")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-crops", action="store_true", help="skip crop/contact-sheet output")
    p.add_argument("--dark-ratio-max", type=float, help="override for this run")
    p.add_argument("--flip-marks-min", type=float, help="override for this run")
    args = p.parse_args()

    overrides = {}
    if args.dark_ratio_max is not None:
        overrides["dark_ratio_max"] = args.dark_ratio_max
    if args.flip_marks_min is not None:
        overrides["flip_marks_min"] = args.flip_marks_min
    cfg = merge_qc_config(overrides)

    if args.selftest:
        raise SystemExit(run_selftest(cfg))

    if args.suggest:
        paths = sorted({p for pat in args.suggest for p in (glob.glob(pat) or [pat])})
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"❌ no such file(s): {missing}")
        raise SystemExit(suggest(paths, cfg))

    if not (args.weights and args.source and args.label):
        p.error("a feature dump needs --weights, --source and --label "
                "(or use --selftest / --suggest)")

    dump_features(
        weights=args.weights, source=args.source, label=args.label, out_dir=args.out,
        cfg=cfg, class_name=args.class_name, conf=args.conf, imgsz=args.imgsz,
        device=args.device, save_crops=not args.no_crops,
    )


if __name__ == "__main__":
    main()
