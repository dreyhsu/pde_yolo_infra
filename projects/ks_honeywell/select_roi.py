"""Select a counting rectangle with OpenCV and save original-frame coordinates."""

import argparse
from pathlib import Path
import sys

from inference_manual_pickup import save_roi


def selection_to_source(selection, preview_size, source_size):
    x, y, width, height = selection
    if width <= 0 or height <= 0:
        return None
    preview_width, preview_height = preview_size
    source_width, source_height = source_size
    return (max(0, round(x * source_width / preview_width)),
            max(0, round(y * source_height / preview_height)),
            min(source_width, round((x + width) * source_width / preview_width)),
            min(source_height, round((y + height) * source_height / preview_height)))


def select_from_video(source, output, frame_index=0, *, cv2_module=None):
    if not Path(source).is_file():
        raise ValueError(f"Source video not found: {source}")
    if Path(output).suffix.lower() != ".json" or Path(output).resolve() == Path(source).resolve():
        raise ValueError("ROI output must be a separate .json file")
    if frame_index < 0:
        raise ValueError("Frame index must be nonnegative")
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {source}")
        if frame_index and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError(f"Cannot seek to frame {frame_index}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read frame {frame_index}")
    finally:
        cap.release()

    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))
    preview = cv2.resize(frame, (preview_width, preview_height)) if scale < 1 else frame
    window = "Counting ROI - drag rectangle, Enter/Space save, C cancel"
    print("Drag a rectangle. Enter/Space confirms it; C cancels without changing the saved ROI.")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        selected = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(window)
    box = selection_to_source(selected, (preview_width, preview_height), (width, height))
    if box is None:
        print("Selection cancelled; saved ROI unchanged.")
        return False
    save_roi(output, box, width, height)
    print(f"Saved ROI {box} for {width}x{height} video to {output}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "roi.json"))
    parser.add_argument("--frame", type=int, default=0, help="Zero-based preview frame index")
    args = parser.parse_args()
    try:
        return 0 if select_from_video(args.source, args.output, args.frame) else 1
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
