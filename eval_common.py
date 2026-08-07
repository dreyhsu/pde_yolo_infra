"""
Shared helpers for the eval_* pipeline.

This is the only library module in the repo (every other script is standalone).
It exists because eval_label_timeline.py and eval_report.py MUST agree exactly on
frame indexing, ROI filtering and class-name resolution -- duplicating that logic
would let the labeller and the scorer drift apart silently, which would quietly
invalidate every number the pipeline produces.

No __main__ guard: import only.
"""

import csv
import gzip
import json
import os
from datetime import datetime

import numpy as np

SCHEMA_VERSION = 1
NONE_CLASS = "__none__"

# Version of the detection-cache format (DET_COLUMNS + meta.json), kept SEPARATE from
# the project-config SCHEMA_VERSION on purpose. load_project() hard-fails on a config
# whose schema_version does not match, so folding cache-format changes into it would
# force every user to hand-edit config.json just because a new column was cached.
# Bump this when DET_COLUMNS changes; is_cache_fresh() then re-runs inference.
#   2 -- added qc_dark_ratio / qc_mark_frac (screen_qc.py)
DET_SCHEMA_VERSION = 2

# Colours carried over from generate_plotly_chart.py so charts stay recognisable.
LEGACY_COLOR_MAP = {
    "teminal": "#ff7f0e",
    "ipi_card": "#d62728",
    "cable_tie": "#8c564b",
    "tether": "#e377c2",
    "mount_puck": "#bcbd22",
    "usb_cable": "#17becf",
    "mount": "#9467bd",
    "sticker1": "#2ca02c",
    "sticker2": "#20026b",
    "gap_idle": "#4682B4",
}

# Deterministic fallback palette (plotly qualitative.Dark24, inlined so this
# module does not need plotly -- the labeller imports it and plotly is a heavy
# import for an OpenCV tool).
FALLBACK_PALETTE = [
    "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
    "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
    "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
    "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038",
]

# Chart lanes for the screen_qc verdicts are named "<class>:<verdict>", e.g.
# "screeninbox:flipped". They sit ALONGSIDE the plain detection lane rather than
# replacing it: the detector answers "is a screen in the tray", which stays true for a
# flipped or bare one, while the lanes answer whether it was placed correctly.
QC_LANE_SEP = ":"

# Green passes, red is the bare-screen defect, purple the flipped one. Kept out of
# FALLBACK_PALETTE so a QC lane can never collide with a detection class colour.
QC_LANE_COLORS = {"ok": "#2ca02c", "no_plastic": "#d62728", "flipped": "#9467bd"}

DEFAULT_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "_comment": "Use forward slashes in Windows paths, e.g. D:/Dre/PDE_yolo_infra/...",
    "project": "",
    "classes_file": "classes.txt",
    "cache_conf": 0.01,
    "conf_sweep": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "imgsz": 640,
    "device": "cuda",
    "iou_nms": 0.7,
    "max_det": 300,
    "stability_window": 3,
    "merge_gap_frames": 2,
    "min_segment_frames": 3,
    "seg_iou_thresholds": [0.1, 0.25, 0.5],
    "score_weights": {"frame_macro_f1": 0.5, "seg_f1_025": 0.5},
    "class_alias": {},
    # Optional OpenCV OK/NG stage for one class (see screen_qc.py). Off unless a
    # project opts in, so no existing project changes behaviour. hph_hw sets:
    #   {"enabled": true, "class": "screeninbox", "min_conf": 0.25}
    # `min_conf` exists because cache_conf is 0.01 -- without it the pixel pass would
    # run on every junk box in every frame. `thresholds` overrides screen_qc.DEFAULT_QC;
    # leave it empty and re-threshold offline instead, since the cache stores the raw
    # features rather than a verdict.
    "screen_qc": {
        "enabled": False,
        "class": "screeninbox",
        "min_conf": 0.25,
        "thresholds": {},
    },
    "models": [],
    "videos": [],
}


# --------------------------------------------------------------------------
# Class names
# --------------------------------------------------------------------------

def normalize(name):
    """Lowercase and strip separators. Same semantics as data_merge_datasets.py."""
    return str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def read_classes(path):
    """Read a classes.txt (one class per line, blanks skipped)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_classes(project_dir, cfg):
    path = os.path.join(project_dir, cfg.get("classes_file", "classes.txt"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"classes file not found: {path}")
    classes = read_classes(path)
    if not classes:
        raise ValueError(f"classes file is empty: {path}")
    dupes = {c for c in classes if classes.count(c) > 1}
    if dupes:
        raise ValueError(f"duplicate classes in {path}: {sorted(dupes)}")
    return classes


def resolve_class(raw, classes, alias=None):
    """
    Resolve a user-typed class name to a canonical class from classes.txt.

    Tries, in order: bare index, exact match, normalized match, alias table.
    Returns None if unresolvable -- callers must re-prompt rather than guess.
    """
    raw = str(raw).strip()
    if not raw:
        return None

    if raw.isdigit():
        idx = int(raw)
        return classes[idx] if 0 <= idx < len(classes) else None

    if raw in classes:
        return raw

    norm_lookup = {normalize(c): c for c in classes}
    if normalize(raw) in norm_lookup:
        return norm_lookup[normalize(raw)]

    for key, target in (alias or {}).items():
        if normalize(key) == normalize(raw):
            return norm_lookup.get(normalize(target))

    return None


def build_class_id_map(model_names, classes, alias=None):
    """
    Map a model's native class ids onto project class indices.

    model_names: {"0": "screen", ...} as stored in the cache meta.
    Returns (id_to_k, unmapped_model, missing_project) so the caller can report
    the mapping instead of silently dropping detections.
    """
    id_to_k = {}
    unmapped_model = []
    for raw_id, raw_name in model_names.items():
        canonical = resolve_class(raw_name, classes, alias)
        if canonical is None:
            unmapped_model.append((int(raw_id), raw_name))
        else:
            id_to_k[int(raw_id)] = classes.index(canonical)
    covered = {classes[k] for k in id_to_k.values()}
    missing_project = [c for c in classes if c not in covered]
    return id_to_k, unmapped_model, missing_project


def build_color_map(classes):
    """Stable class -> #rrggbb. QC lanes and legacy colours win; the rest come from the
    palette by index."""
    color_map = {}
    used = 0
    for cls in classes:
        lane = cls.split(QC_LANE_SEP, 1)[1] if QC_LANE_SEP in cls else None
        if lane in QC_LANE_COLORS:
            # `used` deliberately does not advance here: turning QC lanes on must not
            # shift the palette colours of the detection classes underneath them.
            color_map[cls] = QC_LANE_COLORS[lane]
            continue
        legacy = LEGACY_COLOR_MAP.get(normalize(cls))
        if legacy:
            color_map[cls] = legacy
        else:
            color_map[cls] = FALLBACK_PALETTE[used % len(FALLBACK_PALETTE)]
            used += 1
    return color_map


# --------------------------------------------------------------------------
# Project config
# --------------------------------------------------------------------------

def _norm_path(p):
    return os.path.normpath(str(p)) if p else p


def load_project(project_dir):
    """Load and validate projects/<name>/config.json."""
    cfg_path = os.path.join(project_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.json not found in {project_dir}. Run eval_init_project.py first."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version {cfg.get('schema_version')} != expected {SCHEMA_VERSION}"
        )

    for key, default in DEFAULT_CONFIG.items():
        cfg.setdefault(key, default)

    cache_conf = float(cfg["cache_conf"])
    for c in cfg["conf_sweep"]:
        if float(c) < cache_conf:
            raise ValueError(
                f"conf_sweep value {c} is below cache_conf {cache_conf}. The detection "
                f"cache does not contain those boxes, so the sweep would be silently "
                f"truncated. Lower cache_conf and re-run eval_run_detect.py --force."
            )

    seen = set()
    for m in cfg["models"]:
        if not m.get("name") or not m.get("weights"):
            raise ValueError(f"model entry needs 'name' and 'weights': {m}")
        if m["name"] in seen:
            raise ValueError(f"duplicate model name: {m['name']}")
        seen.add(m["name"])
        m["weights"] = _norm_path(m["weights"])

    seen = set()
    for v in cfg["videos"]:
        if not v.get("name") or not v.get("path"):
            raise ValueError(f"video entry needs 'name' and 'path': {v}")
        if v["name"] in seen:
            raise ValueError(f"duplicate video name: {v['name']}")
        seen.add(v["name"])
        v["path"] = _norm_path(v["path"])

    cfg["_project_dir"] = project_dir
    return cfg


def save_project(project_dir, cfg):
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(os.path.join(project_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def model_conf(cfg, model_entry, key):
    """Per-model override falling back to the project-level value."""
    return model_entry.get(key, cfg.get(key))


def sanitize(name):
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def gt_path(project_dir, video_name):
    return os.path.join(project_dir, "gt", f"{sanitize(video_name)}.gt.json")


def new_gt(video_name, video_path, fps, total_frames, width, height):
    return {
        "schema_version": SCHEMA_VERSION,
        "time_unit": "frames",
        "video_name": video_name,
        "video_path": str(video_path).replace("\\", "/"),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
        "rois": [],
        "segments": [],
    }


def load_gt(project_dir, video_name):
    path = gt_path(project_dir, video_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # Frames are the single source of truth. A hand-edit to seconds would shift
    # every segment against every prediction, so refuse to guess.
    if gt.get("time_unit") != "frames":
        raise ValueError(
            f"{path}: time_unit is {gt.get('time_unit')!r}, expected 'frames'. "
            f"Segment boundaries must be 0-based inclusive frame indices."
        )
    if gt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {gt.get('schema_version')} != {SCHEMA_VERSION}")

    roi_names = {r["name"] for r in gt.get("rois", [])}
    for seg in gt.get("segments", []):
        if seg["roi"] not in roi_names:
            raise ValueError(f"{path}: segment {seg.get('id')} references unknown ROI {seg['roi']!r}")
        if seg["start_frame"] > seg["end_frame"]:
            raise ValueError(f"{path}: segment {seg.get('id')} has start > end")
    return gt


def save_gt(project_dir, gt):
    os.makedirs(os.path.join(project_dir, "gt"), exist_ok=True)
    gt["labeled_at"] = datetime.now().isoformat(timespec="seconds")
    path = gt_path(project_dir, gt["video_name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    return path


def get_roi(gt, roi_name):
    for roi in gt.get("rois", []):
        if roi["name"] == roi_name:
            return roi
    raise KeyError(f"ROI {roi_name!r} not found in {gt.get('video_name')}")


def gt_frame_counts(gt, roi_name, classes):
    """
    Expand GT segments into a dense (total_frames, n_classes) count array.

    Overlapping segments of the same class in the same ROI sum their counts --
    legal but usually a mistake, so the caller gets a warning list back.
    """
    total = int(gt["total_frames"])
    counts = np.zeros((total, len(classes)), dtype=np.int32)
    index = {c: i for i, c in enumerate(classes)}
    warnings = []

    for seg in gt["segments"]:
        if seg["roi"] != roi_name:
            continue
        if seg["class"] not in index:
            warnings.append(f"segment {seg.get('id')}: class {seg['class']!r} not in classes.txt")
            continue
        k = index[seg["class"]]
        s = max(0, int(seg["start_frame"]))
        e = min(total - 1, int(seg["end_frame"]))
        if s > e:
            continue
        if counts[s:e + 1, k].any():
            warnings.append(
                f"segment {seg.get('id')} ({seg['class']}) overlaps another segment of the "
                f"same class in ROI {roi_name!r}; counts will sum"
            )
        counts[s:e + 1, k] += int(seg.get("count", 1))

    return counts, warnings


def gt_segments_for(gt, roi_name, class_name):
    """GT segments for one (roi, class), sorted by start frame, as (start, end, count)."""
    segs = [
        (int(s["start_frame"]), int(s["end_frame"]), int(s.get("count", 1)))
        for s in gt["segments"]
        if s["roi"] == roi_name and s["class"] == class_name
    ]
    return sorted(segs)


# --------------------------------------------------------------------------
# Detection cache
# --------------------------------------------------------------------------

# qc_* hold screen_qc features, not verdicts, and are blank on every row the QC stage
# did not measure. Caching features keeps the thresholds sweepable offline -- the same
# reason detections are cached at cache_conf and swept over conf_sweep later.
DET_COLUMNS = ["frame", "cls_id", "conf", "x1", "y1", "x2", "y2",
               "qc_dark_ratio", "qc_mark_frac"]


def cache_paths(project_dir, model_name, video_name):
    stem = f"{sanitize(model_name)}__{sanitize(video_name)}"
    base = os.path.join(project_dir, "cache", stem)
    return base + ".det.csv.gz", base + ".meta.json"


def open_det_writer(path):
    """Open a gzip CSV detection writer with the header already written."""
    fh = gzip.open(path, "wt", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(DET_COLUMNS)
    return fh, writer


def load_det_cache(project_dir, model_name, video_name):
    """Return (detections DataFrame, meta dict). Raises if the cache is absent."""
    import pandas as pd

    det_path, meta_path = cache_paths(project_dir, model_name, video_name)
    if not os.path.exists(det_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"no detection cache for {model_name} x {video_name}. Run eval_run_detect.py."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(det_path)
    return df, meta


def box_center_in_roi(cx, cy, roi):
    """Locked decision: a detection counts iff its box centre is inside the ROI rect."""
    return (roi["x1"] <= cx <= roi["x2"]) and (roi["y1"] <= cy <= roi["y2"])


def pred_frame_counts(df, meta, roi, classes, conf_thr, total_frames, alias=None):
    """
    Dense per-frame prediction arrays for one (model, video, roi, conf).

    Returns:
      counts (T, K) int32   -- instances per class per frame inside the ROI
      confs  (T, K) float32 -- max confidence per class per frame (0 where absent)
      stats  dict           -- mapping diagnostics for the report

    Confidence is aggregated with max, not mean: a duplicate low-confidence box on
    the same object would otherwise drag the frame's score down and make the Gantt
    bar look dimmer than the model actually was.
    """
    K = len(classes)
    counts = np.zeros((total_frames, K), dtype=np.int32)
    confs = np.zeros((total_frames, K), dtype=np.float32)

    id_to_k, unmapped_model, missing_project = build_class_id_map(
        meta.get("model_names", {}), classes, alias
    )
    stats = {
        "unmapped_model_classes": unmapped_model,
        "project_classes_not_in_model": missing_project,
        "rows_total": int(len(df)),
        "rows_dropped_unmapped": 0,
        "rows_dropped_outside_roi": 0,
        "rows_dropped_below_conf": 0,
        "rows_dropped_out_of_range": 0,
        "rows_kept": 0,
    }

    if len(df) == 0:
        return counts, confs, stats

    frame = df["frame"].to_numpy(dtype=np.int64)
    cls_id = df["cls_id"].to_numpy(dtype=np.int64)
    conf = df["conf"].to_numpy(dtype=np.float32)
    cx = (df["x1"].to_numpy(dtype=np.float64) + df["x2"].to_numpy(dtype=np.float64)) / 2.0
    cy = (df["y1"].to_numpy(dtype=np.float64) + df["y2"].to_numpy(dtype=np.float64)) / 2.0

    keep_conf = conf >= float(conf_thr)
    stats["rows_dropped_below_conf"] = int((~keep_conf).sum())

    mapped = np.full(len(df), -1, dtype=np.int64)
    for raw_id, k in id_to_k.items():
        mapped[cls_id == raw_id] = k
    keep_cls = mapped >= 0
    stats["rows_dropped_unmapped"] = int((keep_conf & ~keep_cls).sum())

    keep_roi = (
        (cx >= roi["x1"]) & (cx <= roi["x2"]) & (cy >= roi["y1"]) & (cy <= roi["y2"])
    )
    stats["rows_dropped_outside_roi"] = int((keep_conf & keep_cls & ~keep_roi).sum())

    keep_range = (frame >= 0) & (frame < total_frames)
    stats["rows_dropped_out_of_range"] = int((keep_conf & keep_cls & keep_roi & ~keep_range).sum())

    keep = keep_conf & keep_cls & keep_roi & keep_range
    stats["rows_kept"] = int(keep.sum())
    if not keep.any():
        return counts, confs, stats

    f = frame[keep]
    k = mapped[keep]
    c = conf[keep]
    np.add.at(counts, (f, k), 1)
    np.maximum.at(confs, (f, k), c)

    return counts, confs, stats


# --------------------------------------------------------------------------
# screen_qc chart lanes
# --------------------------------------------------------------------------

def qc_settings(cfg):
    """(qc_class, thresholds) when the project enables screen_qc, else (None, None)."""
    block = cfg.get("screen_qc") or {}
    if not block.get("enabled"):
        return None, None
    return block.get("class", "screeninbox"), block.get("thresholds") or {}


def qc_lane_names(cfg, classes):
    """
    Lane names to append to `classes` for charting, or [] when screen_qc is off.

    screen_qc is imported lazily throughout this section: it needs cv2, and eval_report.py
    otherwise has no reason to require OpenCV just to draw a chart.
    """
    qc_class, _ = qc_settings(cfg)
    if not qc_class or qc_class not in classes:
        return []
    from screen_qc import QC_DEFECTS, QC_OK

    return [f"{qc_class}{QC_LANE_SEP}{v}" for v in (QC_OK,) + tuple(QC_DEFECTS)]


def qc_chart_lanes(cfg, classes, df, meta, roi, conf_thr, total_frames, alias=None):
    """
    {lane_name: [(start, end, mean_conf), ...]} for the screen_qc verdicts, or {} when
    the stage is off or the cache predates it.

    Verdicts come from the qc_* feature columns, thresholded HERE rather than at cache
    time, so retuning dark_ratio_max/flip_marks_min only costs a re-render.

    Filtering mirrors pred_frame_counts() exactly -- confidence, ROI by box centre,
    frame range -- so a QC lane can never claim a frame the detection lane does not.
    Where a frame holds several boxes of the class the highest-confidence one wins: the
    station holds one tray at a time, same rule inference_screen_qc.py uses.
    """
    lanes = qc_lane_names(cfg, classes)
    if not lanes:
        return {}
    if "qc_dark_ratio" not in getattr(df, "columns", ()):
        return {}  # cache written before screen_qc existed; re-run eval_run_detect.py

    from screen_qc import QC_UNKNOWN, classify_features, merge_qc_config

    qc_class, thresholds = qc_settings(cfg)
    qc_cfg = merge_qc_config(thresholds)

    # Reuse the detection lane's own id mapping rather than matching names again --
    # that is what guarantees the lanes and the bar above them describe the same boxes.
    id_to_k, _, _ = build_class_id_map(meta.get("model_names", {}), classes, alias)
    k_qc = classes.index(qc_class)
    ids = [raw for raw, k in id_to_k.items() if k == k_qc]

    verdicts = np.full(total_frames, QC_UNKNOWN, dtype=object)
    best_conf = np.zeros(total_frames, dtype=np.float32)

    if len(df) and ids:
        frame = df["frame"].to_numpy(dtype=np.int64)
        cls_id = df["cls_id"].to_numpy(dtype=np.int64)
        conf = df["conf"].to_numpy(dtype=np.float32)
        cx = (df["x1"].to_numpy(dtype=np.float64) + df["x2"].to_numpy(dtype=np.float64)) / 2.0
        cy = (df["y1"].to_numpy(dtype=np.float64) + df["y2"].to_numpy(dtype=np.float64)) / 2.0
        dark = df["qc_dark_ratio"].to_numpy(dtype=np.float64)
        mark = df["qc_mark_frac"].to_numpy(dtype=np.float64)

        keep = (conf >= float(conf_thr)) & np.isin(cls_id, ids)
        keep &= (cx >= roi["x1"]) & (cx <= roi["x2"]) & (cy >= roi["y1"]) & (cy <= roi["y2"])
        keep &= (frame >= 0) & (frame < total_frames)

        idx = np.flatnonzero(keep)
        # Ascending confidence, so the highest-confidence box is written last and wins.
        for i in idx[np.argsort(conf[idx], kind="stable")]:
            feats = None
            if np.isfinite(dark[i]) and np.isfinite(mark[i]):
                feats = {"dark_ratio": float(dark[i]), "mark_frac": float(mark[i])}
            # feats is None where the box sat below screen_qc's min_conf or was
            # unmeasurable -> QC_UNKNOWN, which gets no lane. A detection bar with no
            # lane beneath it means "seen but not judged", not "judged OK".
            verdicts[frame[i]] = classify_features(feats, qc_cfg)
            best_conf[frame[i]] = conf[i]

    merge_gap = cfg["merge_gap_frames"]
    min_len = cfg["min_segment_frames"]
    out = {}
    for lane in lanes:
        mask = verdicts == lane.split(QC_LANE_SEP, 1)[1]
        out[lane] = [
            (s, e, float(best_conf[s:e + 1][best_conf[s:e + 1] > 0].mean())
             if (best_conf[s:e + 1] > 0).any() else 0.0)
            for s, e in smooth_segments(rle_segments(mask), merge_gap, min_len)
        ]
    return out


# --------------------------------------------------------------------------
# Temporal segment helpers
# --------------------------------------------------------------------------

def rle_segments(binary):
    """
    Run-length encode a boolean array into inclusive [start, end] runs of True.

    Generalised from the array_to_df loop in generate_plotly_chart.py:82.
    """
    arr = np.asarray(binary, dtype=bool)
    if arr.size == 0 or not arr.any():
        return []
    padded = np.concatenate(([False], arr, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e - 1)) for s, e in zip(edges[0::2], edges[1::2])]


def smooth_segments(segs, merge_gap=2, min_len=3):
    """Close gaps of <= merge_gap frames, then drop runs shorter than min_len."""
    if not segs:
        return []
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s + 1) >= min_len]


def temporal_iou(a, b):
    """Inclusive-frame temporal IoU of two (start, end) segments."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def levenshtein(a, b):
    """Edit distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def frames_to_timecode(frame, fps):
    """Frame index -> mm:ss.ff for HUD and report display."""
    if not fps or fps <= 0:
        return "--:--.--"
    total_sec = frame / float(fps)
    return f"{int(total_sec // 60):02d}:{total_sec % 60:05.2f}"
