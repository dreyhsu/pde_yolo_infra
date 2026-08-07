"""
Run YOLO + the screen_qc OpenCV check over a video and report the OK/NG episodes.

The detector alone cannot do this: it scores ~0.93 on a good placement, on a bare screen
and on a flipped one alike (transfer/screen_ng/). screen_qc.py reads the pixels inside
each `screeninbox` box and separates them.

    python inference_screen_qc.py --weights logs/hph_hw/yolo11n/weights/best.pt \
        --source video/hph_honeywell/test.mp4 --save out.mp4

    python inference_screen_qc.py --weights best.pt --source clip.mp4 --show
    python inference_screen_qc.py --weights best.pt --source clip.mp4 \
        --dark-ratio-max 0.6 --flip-marks-min 0.18     # try thresholds without editing

A frame's verdict comes from its highest-confidence `screeninbox` box -- the station holds
one tray at a time. Verdicts are then majority-voted over `--window` frames so one
specular highlight cannot flip the call; the vote is centred, so annotated output lags the
input by window//2 frames and only that many frames are ever held in memory.

Thresholds live in screen_qc.DEFAULT_QC. Fit them to your line with
calibrate_screen_qc.py before trusting the NG counts here.
"""

import argparse
import os
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

from eval_common import frames_to_timecode, rle_segments, smooth_segments
from screen_qc import (
    QC_DEFECTS,
    QC_UNKNOWN,
    QC_VERDICTS,
    VERDICT_LABEL,
    classify_screeninbox,
    draw_verdict,
    majority_verdict,
    merge_qc_config,
)


def target_ids(model_names, class_name):
    """Model class ids whose name matches `class_name`, ignoring case and separators."""
    def norm(s):
        return str(s).strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    target = norm(class_name)
    return {int(k) for k, v in model_names.items() if norm(v) == target}


def frame_detections(result, want_ids, qc_cfg):
    """
    [(box, conf, verdict, feats)] for this frame's boxes of the wanted class, highest
    confidence first. The head of the list is the frame's primary tray.
    """
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    out = []
    for box, c, p in zip(xyxy, cls, conf):
        if int(c) not in want_ids:
            continue
        verdict, feats = classify_screeninbox(result.orig_img, box, qc_cfg)
        out.append((box, float(p), verdict, feats))
    out.sort(key=lambda d: -d[1])
    return out


def draw_frame(frame, dets, smoothed):
    """Primary box gets the voted verdict; any extra boxes keep their own raw one."""
    for i, (box, _conf, raw, feats) in enumerate(dets):
        verdict = smoothed if i == 0 else raw
        draw_verdict(frame, box, verdict, feats, thickness=2 if i == 0 else 1)
    return frame


def report_episodes(timeline, raw_timeline, fps, merge_gap, min_len, class_name):
    """Collapse the per-frame verdicts into defect episodes with timecodes."""
    print("\nframe verdicts (after the vote)")
    total = max(len(timeline), 1)
    for v in QC_VERDICTS:
        n = timeline.count(v)
        if n:
            print(f"   {VERDICT_LABEL[v]:<14} {n:6d}  ({100.0 * n / total:5.1f}%)")

    # The vote lets a neighbour's verdict cover an unmeasurable frame, which is what
    # keeps occlusion from punching holes in a defect run -- but it also means frames
    # with no tray at all get absorbed into whatever surrounds them. Say so plainly.
    n_inherited = sum(
        1 for raw, voted in zip(raw_timeline, timeline)
        if raw == QC_UNKNOWN and voted != QC_UNKNOWN
    )
    if n_inherited:
        print(f"   ({n_inherited} of those frames had no measurable {class_name} of "
              f"their own and inherited a neighbour's verdict)")

    episodes = []
    for defect in QC_DEFECTS:
        segs = smooth_segments(
            rle_segments([v == defect for v in timeline]), merge_gap, min_len
        )
        episodes += [(s, e, defect) for s, e in segs]
    episodes.sort()

    if not episodes:
        print("\n✅ no NG episodes")
        return episodes

    print(f"\n❌ {len(episodes)} NG episode(s)  "
          f"(runs of >= {min_len} frames, gaps of <= {merge_gap} closed)")
    for s, e, defect in episodes:
        print(f"   {frames_to_timecode(s, fps)} -> {frames_to_timecode(e, fps)}  "
              f"frames {s}-{e} ({e - s + 1})  {VERDICT_LABEL[defect]}")
    return episodes


def run(weights, source, out_path=None, show=False, class_name="screeninbox",
        conf=0.25, imgsz=640, device="cuda", window=5, qc_cfg=None,
        merge_gap=2, min_len=3):
    from ultralytics import YOLO

    qc_cfg = qc_cfg if qc_cfg is not None else merge_qc_config()
    model = YOLO(weights)
    names = {str(k): v for k, v in model.names.items()}
    want = target_ids(names, class_name)
    if not want:
        raise SystemExit(
            f"model has no class named {class_name!r}; it has {sorted(names.values())}"
        )

    cap = cv2.VideoCapture(source)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()

    writer = None
    timeline = []          # voted verdict per frame, in frame order
    raw_timeline = []      # pre-vote verdict per frame, for the summary only
    half = max(int(window), 1) // 2

    # At most 2*half+1 frames are ever resident -- enough to centre the vote, and nowhere
    # near the whole video. Same reason eval_run_detect.py insists on stream=True.
    buf = deque()          # (frame, dets, raw) for frames [base .. base+len(buf)-1]
    base = 0               # frame index of buf[0]
    pending = 0            # next frame index still to emit

    def emit(frame, dets, verdict):
        nonlocal writer
        if not (out_path or show):
            return
        draw_frame(frame, dets, verdict)
        if out_path:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h)
                )
                if not writer.isOpened():
                    raise IOError(f"could not open video writer for {out_path}")
            writer.write(frame)
        if show:
            cv2.imshow("screen_qc", frame)
            cv2.waitKey(1)

    def flush_one():
        """Emit frame `pending` using the widest centred window the buffer can supply."""
        nonlocal pending, base
        lo = max(pending - half, base)
        hi = min(pending + half, base + len(buf) - 1)
        frame, dets, own = buf[pending - base]
        verdict = majority_verdict(
            [buf[k - base][2] for k in range(lo, hi + 1)], prefer=own
        )
        timeline.append(verdict)
        emit(frame, dets, verdict)
        pending += 1
        while base < pending - half:   # drop frames no later window can need
            buf.popleft()
            base += 1

    results = model.predict(
        source=source, conf=conf, imgsz=imgsz, device=device,
        save=False, show=False, stream=True, verbose=False,
    )
    n_frames = 0
    try:
        for i, r in enumerate(results):
            n_frames = i + 1
            dets = frame_detections(r, want, qc_cfg)
            raw = dets[0][2] if dets else QC_UNKNOWN
            raw_timeline.append(raw)
            # orig_img is a view into the Results object, which is freed each iteration.
            buf.append((r.orig_img.copy() if (out_path or show) else None, dets, raw))
            # Frame `pending` can be emitted once the right half of its window has
            # arrived. The first `half` frames wait until frame 2*half, then flush
            # together with left-clipped windows -- so no frame is ever skipped.
            while pending + half <= i:
                flush_one()
            del r

        # Tail: nothing more is coming, so the last windows are right-clipped.
        while pending < n_frames:
            flush_one()
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n{len(timeline)} frames, fps={fps:.2f}, vote window={window}")
    episodes = report_episodes(
        timeline, raw_timeline, fps, merge_gap, min_len, class_name
    )
    if out_path:
        print(f"\n💾 annotated video -> {out_path}")
    return timeline, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="video file")
    p.add_argument("--save", dest="out_path", help="write an annotated mp4 here")
    p.add_argument("--show", action="store_true", help="live preview window")
    p.add_argument("--class-name", default="screeninbox")
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--window", type=int, default=5, help="majority-vote window (frames)")
    p.add_argument("--merge-gap", type=int, default=2,
                   help="close NG gaps of at most this many frames")
    p.add_argument("--min-len", type=int, default=3,
                   help="ignore NG runs shorter than this")
    p.add_argument("--dark-ratio-max", type=float, help="override the no_plastic threshold")
    p.add_argument("--flip-marks-min", type=float, help="override the flipped threshold")
    args = p.parse_args()

    overrides = {}
    if args.dark_ratio_max is not None:
        overrides["dark_ratio_max"] = args.dark_ratio_max
    if args.flip_marks_min is not None:
        overrides["flip_marks_min"] = args.flip_marks_min

    run(
        weights=args.weights, source=args.source, out_path=args.out_path, show=args.show,
        class_name=args.class_name, conf=args.conf, imgsz=args.imgsz, device=args.device,
        window=args.window, qc_cfg=merge_qc_config(overrides),
        merge_gap=args.merge_gap, min_len=args.min_len,
    )


if __name__ == "__main__":
    main()
