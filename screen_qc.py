"""
OpenCV OK/NG check for the `screeninbox` class (project hph_hw).

The detector answers "is there a screen in the tray?" and nothing more. It fires at
~0.93 on all three of transfer/screen_ng/{normal,plastic_ng,flip_ng}.png, so no
confidence threshold can separate a good placement from the two defects:

    no_plastic -- screen dropped in bare, without its protective plastic bag
    flipped    -- screen placed back-side up

Both defects DO show up in the pixels inside the box, which is what this module reads.

Two features, one per defect, each deliberately illumination-invariant so they survive
lamp and exposure changes on the line:

  dark_ratio = median(core) / median(tray ring)
      Bare screen glass is a black mirror: the core collapses (median 5) while the
      cardboard tray around it stays lit (median 37) -> ratio 0.135. The plastic bag
      scatters light, so a bagged screen is BRIGHTER than the tray -> ratio > 1.
      Dividing by the tray ring is what makes this exposure-proof; a raw brightness
      threshold would not survive a lamp change.

  mark_frac = fraction of core pixels more than `mark_contrast` above their own
      Gaussian-blurred background.
      The back of the panel carries printed labels, barcodes and the flex-cable
      cutout. Subtracting the local background isolates exactly that printing and
      throws away overall brightness -> 0.263 flipped vs ~0.05 front-facing.

Measured on the three reference images (the defaults below sit in those gaps).
`python calibrate_screen_qc.py --selftest` reproduces this table exactly:

    feature       normal   plastic_ng   flip_ng
    dark_ratio     1.314      0.135       1.189
    mark_frac      0.055      0.045       0.263
    lc_std        15.3       13.3        42.1     (diagnostic only)
    grad_mean     40.1       29.2       107.0     (diagnostic only)

Feature extraction and thresholding are SEPARATE on purpose. The pixel pass is the
expensive part and can only run while the frame is in memory; the thresholds are cheap
and must stay tunable. Caching features and thresholding later mirrors the
cache_conf / conf_sweep split eval_run_detect.py already relies on -- retuning a
threshold must never cost another GPU pass.

cv2 + numpy only. No ultralytics, no __main__ -- import only, same shape as
eval_common.py. `python calibrate_screen_qc.py --selftest` exercises it.
"""

import cv2
import numpy as np

QC_OK = "ok"
QC_NO_PLASTIC = "no_plastic"
QC_FLIPPED = "flipped"
QC_UNKNOWN = "unknown"

# Order matters for reporting only; QC_UNKNOWN is never a defect claim.
QC_VERDICTS = (QC_OK, QC_NO_PLASTIC, QC_FLIPPED, QC_UNKNOWN)
QC_DEFECTS = (QC_NO_PLASTIC, QC_FLIPPED)

# Written to the detection cache by eval_run_detect.py, one column each.
FEATURE_COLUMNS = ("dark_ratio", "mark_frac", "lc_std", "grad_mean")

DEFAULT_QC = {
    # --- crop geometry, as a fraction of the box inset from each edge ---
    "core_inset": 0.30,        # middle 40% of the box = the screen face itself
    "ring_outer_inset": 0.06,  # the 6%..20% annulus is cardboard tray, used as the
    "ring_inner_inset": 0.20,  #   per-frame illumination reference
    "norm_size": [256, 160],   # resize the core so texture features are scale-free

    # --- feature extraction ---
    "blur_sigma": 9.0,         # background scale for the local-contrast image
    "mark_contrast": 25,       # a pixel this far above its background counts as print

    # --- guards: refuse to guess rather than emit a wrong defect call ---
    "min_box_px": 60,          # boxes smaller than this -> QC_UNKNOWN
    "min_ring_median": 8.0,    # a black tray ring makes dark_ratio meaningless

    # --- thresholds (fitted to 3 frames; recalibrate with calibrate_screen_qc.py) ---
    "dark_ratio_max": 1.027,    # below -> no_plastic   (0.135 vs 1.189 / 1.314)
    "flip_marks_min": 0.151,    # above -> flipped      (0.263 vs 0.045 / 0.055)
}


# The only two settings that affect the VERDICT but not the cached features. Everything
# else in DEFAULT_QC changes what gets measured, so changing it must invalidate a cache.
THRESHOLD_KEYS = ("dark_ratio_max", "flip_marks_min")


def extraction_fingerprint(cfg):
    """The subset of `cfg` that changes the measured features.

    eval_run_detect.py stores this in meta.json and re-runs inference when it changes.
    Deliberately excludes THRESHOLD_KEYS: retuning a threshold re-reads the cached
    features and must never cost another GPU pass.
    """
    return {k: v for k, v in sorted(cfg.items()) if k not in THRESHOLD_KEYS}


def merge_qc_config(overrides=None):
    """DEFAULT_QC updated with `overrides`. Unknown keys raise -- a typo in a config
    file must not silently leave a threshold at its default."""
    cfg = dict(DEFAULT_QC)
    if not overrides:
        return cfg
    unknown = [k for k in overrides if k not in DEFAULT_QC and k != "enabled"]
    if unknown:
        raise KeyError(f"unknown screen_qc setting(s): {sorted(unknown)}")
    cfg.update({k: v for k, v in overrides.items() if k != "enabled"})
    return cfg


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _clamp_box(box, width, height):
    """Box as ints clipped to the frame. Returns None if nothing is left."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(width), x2), min(int(height), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _inset(box, frac):
    """Shrink a box by `frac` of its width/height on every side."""
    x1, y1, x2, y2 = box
    dx = int(round(frac * (x2 - x1)))
    dy = int(round(frac * (y2 - y1)))
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _crop(gray, rect):
    x1, y1, x2, y2 = rect
    return gray[y1:y2, x1:x2]


def _ring_pixels(gray, box, cfg):
    """The tray annulus between the outer and inner insets, as a flat array."""
    ox1, oy1, ox2, oy2 = _inset(box, cfg["ring_outer_inset"])
    ix1, iy1, ix2, iy2 = _inset(box, cfg["ring_inner_inset"])
    outer = gray[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return outer.reshape(-1)
    keep = np.ones(outer.shape, dtype=bool)
    # inner rect expressed relative to the outer crop
    keep[max(0, iy1 - oy1):max(0, iy2 - oy1), max(0, ix1 - ox1):max(0, ix2 - ox1)] = False
    return outer[keep]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def screeninbox_features(frame_bgr, box, cfg=None):
    """
    Pixel features for one `screeninbox` detection.

    frame_bgr : full BGR frame (Ultralytics `r.orig_img`)
    box       : (x1, y1, x2, y2) in frame pixels

    Returns a dict with FEATURE_COLUMNS plus `core_median`/`ring_median`, or None when
    the box is too small or too clipped to measure. None means "no opinion" -- callers
    map it to QC_UNKNOWN, never to a defect.
    """
    cfg = cfg if cfg is not None else DEFAULT_QC
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h_img, w_img = frame_bgr.shape[:2]
    clamped = _clamp_box(box, w_img, h_img)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    if min(x2 - x1, y2 - y1) < cfg["min_box_px"]:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    core = _crop(gray, _inset(clamped, cfg["core_inset"]))
    ring = _ring_pixels(gray, clamped, cfg)
    if core.size == 0 or ring.size == 0:
        return None

    core_median = float(np.median(core))
    ring_median = float(np.median(ring))
    if ring_median < cfg["min_ring_median"]:
        # The illumination reference itself is black -- the tray is unlit or the box
        # landed on background. dark_ratio would be noise; say nothing.
        return None

    # Size-normalise before any texture measure, so a screen filmed close up does not
    # score differently from the same screen filmed far away.
    nw, nh = int(cfg["norm_size"][0]), int(cfg["norm_size"][1])
    core_n = cv2.resize(core, (nw, nh), interpolation=cv2.INTER_AREA)

    background = cv2.GaussianBlur(core_n, (0, 0), float(cfg["blur_sigma"]))
    local = core_n.astype(np.float32) - background.astype(np.float32)

    sx = cv2.Sobel(core_n, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(core_n, cv2.CV_32F, 0, 1, ksize=3)

    return {
        "dark_ratio": core_median / ring_median,
        "mark_frac": float((local > float(cfg["mark_contrast"])).mean()),
        "lc_std": float(local.std()),
        "grad_mean": float(np.hypot(sx, sy).mean()),
        "core_median": core_median,
        "ring_median": ring_median,
    }


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def classify_features(feats, cfg=None):
    """
    Verdict from already-extracted features -- cheap, so thresholds stay sweepable
    offline against a cached feature column.

    no_plastic is tested first: a screen that is both bare and flipped reports
    no_plastic, and the darkness cue is the physically stronger of the two.
    """
    cfg = cfg if cfg is not None else DEFAULT_QC
    if not feats:
        return QC_UNKNOWN
    dark_ratio = feats.get("dark_ratio")
    mark_frac = feats.get("mark_frac")
    if dark_ratio is None or mark_frac is None:
        return QC_UNKNOWN
    if not (np.isfinite(dark_ratio) and np.isfinite(mark_frac)):
        return QC_UNKNOWN
    if dark_ratio < cfg["dark_ratio_max"]:
        return QC_NO_PLASTIC
    if mark_frac > cfg["flip_marks_min"]:
        return QC_FLIPPED
    return QC_OK


def classify_screeninbox(frame_bgr, box, cfg=None):
    """(verdict, features) for one detection. features is None when unmeasurable."""
    cfg = cfg if cfg is not None else DEFAULT_QC
    feats = screeninbox_features(frame_bgr, box, cfg)
    return classify_features(feats, cfg), feats


def majority_verdict(verdicts, prefer=None):
    """
    Majority vote over one window of verdicts.

    QC_UNKNOWN never wins -- it only survives where the window holds nothing else. That
    is what keeps a handful of occluded frames from punching holes in an otherwise solid
    defect run; the cost is that a frame with no tray at all inherits its neighbours'
    verdict, which callers should say out loud when they report totals.

    Ties break toward `prefer` (normally the frame's own verdict), then toward the
    first verdict seen. Never toward set iteration order -- that is hash-randomised per
    process, which would make the same video score differently on consecutive runs.

    This is the streaming primitive; smooth_verdicts() is the whole-sequence form.
    """
    counts = {}
    for v in verdicts:
        if v != QC_UNKNOWN:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return QC_UNKNOWN
    best = max(counts.values())
    winners = [v for v in counts if counts[v] == best]  # dict keeps insertion order
    return prefer if prefer in winners else winners[0]


def smooth_verdicts(verdicts, window=3):
    """
    Majority vote over a centred rolling window, so one specular frame cannot flip the
    call. For an already-materialised sequence; inference_screen_qc.py uses
    majority_verdict() directly because it votes as frames stream past.

    `window` is the existing `stability_window` config key (eval_common.DEFAULT_CONFIG).
    """
    verdicts = list(verdicts)
    n = len(verdicts)
    if n == 0 or window <= 1:
        return verdicts

    half = int(window) // 2
    return [
        majority_verdict(verdicts[max(0, i - half):min(n, i + half + 1)], verdicts[i])
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Presentation (shared by the realtime and calibration scripts)
# --------------------------------------------------------------------------

# BGR. Green passes, red is a defect, amber means "could not measure".
VERDICT_COLOR = {
    QC_OK: (60, 200, 60),
    QC_NO_PLASTIC: (60, 60, 235),
    QC_FLIPPED: (200, 60, 235),
    QC_UNKNOWN: (60, 190, 235),
}

VERDICT_LABEL = {
    QC_OK: "OK",
    QC_NO_PLASTIC: "NG no plastic",
    QC_FLIPPED: "NG flipped",
    QC_UNKNOWN: "?",
}


def draw_verdict(frame_bgr, box, verdict, feats=None, thickness=2):
    """Draw the box and its verdict in place. Returns the frame for chaining."""
    clamped = _clamp_box(box, frame_bgr.shape[1], frame_bgr.shape[0])
    if clamped is None:
        return frame_bgr
    x1, y1, x2, y2 = clamped
    color = VERDICT_COLOR.get(verdict, VERDICT_COLOR[QC_UNKNOWN])
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

    text = VERDICT_LABEL.get(verdict, verdict)
    if feats:
        text += f"  d={feats['dark_ratio']:.2f} m={feats['mark_frac']:.2f}"
    font, scale, tt = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, tt)
    ty = y1 - 6 if y1 - th - base - 6 >= 0 else y2 + th + base + 6
    cv2.rectangle(frame_bgr, (x1, ty - th - base), (x1 + tw, ty + base), color, -1)
    cv2.putText(frame_bgr, text, (x1, ty), font, scale, (255, 255, 255), tt, cv2.LINE_AA)
    return frame_bgr
