"""
Cache raw YOLO detections for every model x video pair in a project.

Inference runs ONCE per pair at a very low confidence (cache_conf, default 0.01)
and every surviving box is written to disk. eval_report.py then sweeps confidence
thresholds offline, so changing the threshold never costs another GPU pass.

If the project enables the optional "screen_qc" block, each box of the configured class
also gets the screen_qc.py OpenCV features cached alongside it (qc_dark_ratio /
qc_mark_frac). This loop is the only place in the pipeline where pixels and boxes coexist
-- everything downstream reads the CSV. Features are cached, never verdicts, so OK/NG
thresholds stay re-tunable offline for the same reason confidence does.

    python eval_run_detect.py --project projects/hph_hw
    python eval_run_detect.py --project projects/hph_hw --models yolo11n_e500 --videos test1
    python eval_run_detect.py --project projects/hph_hw --force
"""

import argparse
import json
import os
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

from eval_common import (
    DET_SCHEMA_VERSION,
    SCHEMA_VERSION,
    cache_paths,
    load_classes,
    load_project,
    model_conf,
    open_det_writer,
    resolve_class,
)
from screen_qc import extraction_fingerprint, merge_qc_config, screeninbox_features

FLUSH_EVERY = 2000


def qc_target_ids(model_names, class_name):
    """Model class ids whose name matches `class_name`, ignoring case and separators."""
    def norm(s):
        return str(s).strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    target = norm(class_name)
    return {int(k) for k, v in model_names.items() if norm(v) == target}


def probe_video(path):
    """fps / frame count / resolution straight from the container."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "total_frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def is_cache_fresh(meta_path, weights, video_path, cache_conf, imgsz, qc=None):
    """
    A cache is stale if the weights file, the video, or the inference settings
    changed. Cheap stat()-based check -- no hashing of multi-GB files.

    `qc` is the screen_qc fingerprint (see build_qc_settings): whether the QC stage ran,
    on which class, and with which extraction parameters. Verdict thresholds are NOT in
    it -- they are applied to the cached features offline, so retuning one must not
    trigger another GPU pass.
    """
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if meta.get("det_schema_version") != DET_SCHEMA_VERSION:
        return False
    try:
        w_stat = os.stat(weights)
        v_stat = os.stat(video_path)
    except OSError:
        return False

    return (
        meta.get("weights_size") == w_stat.st_size
        and abs(meta.get("weights_mtime", -1) - w_stat.st_mtime) < 1.0
        and meta.get("video_size") == v_stat.st_size
        and float(meta.get("cache_conf", -1)) == float(cache_conf)
        and int(meta.get("imgsz", -1)) == int(imgsz)
        and meta.get("screen_qc") == qc
    )


def build_qc_settings(cfg):
    """
    Resolve the project's optional screen_qc block into the fingerprint stored in
    meta.json, or None when the stage is off. Returns (settings, qc_cfg).
    """
    block = cfg.get("screen_qc") or {}
    if not block.get("enabled"):
        return None, None
    qc_cfg = merge_qc_config(block.get("thresholds"))
    settings = {
        "class": block.get("class", "screeninbox"),
        "min_conf": float(block.get("min_conf", 0.25)),
        "extraction": extraction_fingerprint(qc_cfg),
    }
    return settings, qc_cfg


def run_model_on_video(
    weights,
    video_path,
    out_csv_gz,
    out_meta_json,
    model_name,
    video_name,
    cache_conf=0.01,
    imgsz=640,
    device="cuda",
    iou_nms=0.7,
    max_det=300,
    flush_every=FLUSH_EVERY,
    qc_settings=None,
    qc_cfg=None,
):
    """Stream inference over a video, appending every detection to a gzip CSV."""
    import ultralytics
    from ultralytics import YOLO

    video_info = probe_video(video_path)
    model = YOLO(weights)
    names = {str(k): v for k, v in model.names.items()}

    # This loop is the only point in the pipeline where pixels and boxes coexist --
    # everything downstream reads the CSV. So the screen_qc pixel pass has to happen
    # here or not at all.
    qc_ids, qc_min_conf = set(), 1.1
    if qc_settings:
        qc_ids = qc_target_ids(names, qc_settings["class"])
        qc_min_conf = qc_settings["min_conf"]
        if not qc_ids:
            print(f"   ⚠ screen_qc is enabled for class {qc_settings['class']!r} but the "
                  f"model has no such class -- no QC features will be cached")

    os.makedirs(os.path.dirname(out_csv_gz), exist_ok=True)
    fh, writer = open_det_writer(out_csv_gz)
    n_rows = 0
    n_qc = 0
    n_qc_unmeasurable = 0
    frames_decoded = 0

    try:
        # stream=True is mandatory. Without it predict() returns a list holding
        # every Results object -- each keeping an image tensor alive -- and a
        # 1300-frame 1080p video will exhaust RAM before it finishes.
        results = model.predict(
            source=video_path,
            conf=cache_conf,
            iou=iou_nms,
            max_det=max_det,
            imgsz=imgsz,
            device=device,
            save=False,
            show=False,
            stream=True,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            frames_decoded = frame_idx + 1
            boxes = r.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                    qc_dark, qc_mark = "", ""
                    # The min_conf gate matters: cache_conf is 0.01, so without it the
                    # pixel pass would run on every junk box in every frame.
                    if int(c) in qc_ids and float(p) >= qc_min_conf:
                        feats = screeninbox_features(
                            r.orig_img, (x1, y1, x2, y2), qc_cfg
                        )
                        if feats:
                            qc_dark = round(feats["dark_ratio"], 6)
                            qc_mark = round(feats["mark_frac"], 6)
                            n_qc += 1
                        else:
                            n_qc_unmeasurable += 1
                    writer.writerow(
                        [frame_idx, int(c), round(float(p), 4),
                         int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
                         qc_dark, qc_mark]
                    )
                    n_rows += 1
            del r  # never retain a Results object past its frame

            if frames_decoded % flush_every == 0:
                fh.flush()
                print(f"   ... {frames_decoded} frames, {n_rows} detections")
    finally:
        fh.close()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "det_schema_version": DET_SCHEMA_VERSION,
        "model_name": model_name,
        "weights": str(weights).replace("\\", "/"),
        "weights_size": os.path.getsize(weights),
        "weights_mtime": os.stat(weights).st_mtime,
        "model_names": names,
        "video_name": video_name,
        "video_path": str(video_path).replace("\\", "/"),
        "video_size": os.path.getsize(video_path),
        "fps": video_info["fps"],
        "total_frames_reported": video_info["total_frames_reported"],
        "frames_decoded": frames_decoded,
        "width": video_info["width"],
        "height": video_info["height"],
        "imgsz": int(imgsz),
        "device": device,
        "cache_conf": float(cache_conf),
        "iou_nms": float(iou_nms),
        "max_det": int(max_det),
        "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
        "n_rows": n_rows,
        "screen_qc": qc_settings,
        "n_qc_measured": n_qc,
        "n_qc_unmeasurable": n_qc_unmeasurable,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if qc_settings:
        print(f"   🔍 screen_qc: {n_qc} {qc_settings['class']} boxes measured, "
              f"{n_qc_unmeasurable} unmeasurable")

    if frames_decoded != video_info["total_frames_reported"]:
        print(
            f"   ⚠ decoded {frames_decoded} frames but the container reported "
            f"{video_info['total_frames_reported']}. CAP_PROP_FRAME_COUNT is an estimate on "
            f"VFR/B-frame MP4s; label ground truth against the decoded count "
            f"({frames_decoded}). Consider a CFR re-encode via prep_convert_video.py."
        )

    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return meta


def report_class_mapping(model_names, classes, alias):
    """Surface class-set mismatches before any labeling time is spent."""
    matched, model_only = [], []
    for raw_id, raw_name in sorted(model_names.items(), key=lambda kv: int(kv[0])):
        canonical = resolve_class(raw_name, classes, alias)
        (matched if canonical else model_only).append((raw_id, raw_name, canonical))
    covered = {c for _, _, c in matched}
    project_only = [c for c in classes if c not in covered]

    print(f"   classes: {len(matched)}/{len(model_names)} model classes map to the project")
    if model_only:
        print(f"   ⚠ model-only (detections will be DROPPED): "
              f"{[n for _, n, _ in model_only]}")
    if project_only:
        print(f"   ⚠ project-only (always counted as misses): {project_only}")


def main():
    parser = argparse.ArgumentParser(description="Cache raw YOLO detections for a project.")
    parser.add_argument("--project", required=True, help="Path to projects/<name>")
    parser.add_argument("--models", help="Comma-separated model names (default: all)")
    parser.add_argument("--videos", help="Comma-separated video names (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-run even if the cache is fresh")
    args = parser.parse_args()

    cfg = load_project(args.project)
    classes = load_classes(args.project, cfg)
    alias = cfg.get("class_alias", {})

    models = cfg["models"]
    videos = cfg["videos"]
    if args.models:
        wanted = {s.strip() for s in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]
    if args.videos:
        wanted = {s.strip() for s in args.videos.split(",")}
        videos = [v for v in videos if v["name"] in wanted]

    if not models or not videos:
        print("❌ Nothing to do -- check models[]/videos[] in config.json and your filters.")
        return

    missing = [m["weights"] for m in models if not os.path.exists(m["weights"])]
    missing += [v["path"] for v in videos if not os.path.exists(v["path"])]
    if missing:
        print("❌ Missing files:")
        for p in missing:
            print(f"   {p}")
        return

    qc_settings, qc_cfg = build_qc_settings(cfg)

    print(f"🚀 {len(models)} model(s) x {len(videos)} video(s), cache_conf={cfg['cache_conf']}")
    if qc_settings:
        print(f"   screen_qc ON for class {qc_settings['class']!r} "
              f"at conf >= {qc_settings['min_conf']}")
    for m in models:
        imgsz = model_conf(cfg, m, "imgsz")
        for v in videos:
            det_path, meta_path = cache_paths(args.project, m["name"], v["name"])
            label = f"{m['name']} x {v['name']}"

            if not args.force and is_cache_fresh(
                meta_path, m["weights"], v["path"], cfg["cache_conf"], imgsz, qc_settings
            ):
                print(f"✅ cache fresh, skipped: {label}")
                continue

            print(f"▶ {label}")
            meta = run_model_on_video(
                weights=m["weights"],
                video_path=v["path"],
                out_csv_gz=det_path,
                out_meta_json=meta_path,
                model_name=m["name"],
                video_name=v["name"],
                cache_conf=cfg["cache_conf"],
                imgsz=imgsz,
                device=model_conf(cfg, m, "device"),
                iou_nms=cfg["iou_nms"],
                max_det=cfg["max_det"],
                qc_settings=qc_settings,
                qc_cfg=qc_cfg,
            )
            report_class_mapping(meta["model_names"], classes, alias)
            print(f"   💾 {meta['frames_decoded']} frames, {meta['n_rows']} detections -> {det_path}")

    print("\n✅ Detection cache up to date.")


if __name__ == "__main__":
    main()
