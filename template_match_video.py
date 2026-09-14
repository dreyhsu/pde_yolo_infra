#!/usr/bin/env python3
"""Run ROI-limited OpenCV template matching and tenon measurement on video.

The output video contains the best-match bounding box and confidence score on
every frame. Matching is performed only inside regions loaded from
``saved_rois.txt``. Accepted matches are cropped and measured with the tenon
detector in ``transfer/measure_tenon_area.py``.

Example:
    conda run -n tf python template_match_video.py template.png input.mp4 output.mp4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print(
        "OpenCV and NumPy are required in this Python environment. "
        "Install them with: python3 -m pip install opencv-python numpy",
        file=sys.stderr,
    )
    raise SystemExit(1)

from transfer.measure_tenon_area import detect_tenon_polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match one template in every video frame and save an annotated video."
        )
    )
    parser.add_argument("template", type=Path, help="Path to the template image")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("output", type=Path, help="Path to the output video")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence needed to mark a match as accepted (default: 0.75)",
    )
    parser.add_argument(
        "--roi-file",
        type=Path,
        default=Path(__file__).resolve().parent / "saved_rois.txt",
        help="CSV file containing x,y,w,h search regions (default: saved_rois.txt)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show frames while processing; press q to stop early",
    )
    return parser.parse_args()


def video_fourcc(output_path: Path) -> int:
    """Select a broadly supported OpenCV codec from the output extension."""
    if output_path.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def load_rois(
    roi_path: Path,
    frame_size: tuple[int, int],
    template_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Load, clip, and validate x,y,w,h search regions from a CSV file."""
    if not roi_path.is_file():
        raise ValueError(f"ROI file not found: {roi_path}")

    frame_width, frame_height = frame_size
    template_width, template_height = template_size
    rois: list[tuple[int, int, int, int]] = []

    with roi_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"x", "y", "w", "h"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("ROI file must have the header: x,y,w,h")

        for line_number, row in enumerate(reader, start=2):
            try:
                x = int(row["x"])
                y = int(row["y"])
                width = int(row["w"])
                height = int(row["h"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid ROI values on line {line_number}: {row}"
                ) from error

            if width <= 0 or height <= 0:
                raise ValueError(f"ROI on line {line_number} must have positive w and h")

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width, x + width)
            y2 = min(frame_height, y + height)
            clipped_width = x2 - x1
            clipped_height = y2 - y1

            if clipped_width < template_width or clipped_height < template_height:
                print(
                    f"Warning: skipping ROI on line {line_number}; its clipped size "
                    f"{clipped_width}x{clipped_height} is smaller than the template",
                    file=sys.stderr,
                )
                continue
            rois.append((x1, y1, clipped_width, clipped_height))

    if not rois:
        raise ValueError("No valid ROI is large enough to contain the template")
    return rois


def find_best_roi_match(
    frame_gray,
    template_gray,
    rois: list[tuple[int, int, int, int]],
) -> tuple[float, tuple[int, int]]:
    """Return the best confidence and full-frame location from all configured ROIs."""
    best_confidence = -1.0
    best_top_left = (0, 0)

    for roi_x, roi_y, roi_width, roi_height in rois:
        search_image = frame_gray[
            roi_y : roi_y + roi_height,
            roi_x : roi_x + roi_width,
        ]
        score_map = cv2.matchTemplate(
            search_image, template_gray, cv2.TM_CCOEFF_NORMED
        )
        _, confidence, _, local_top_left = cv2.minMaxLoc(score_map)
        if confidence > best_confidence:
            best_confidence = float(confidence)
            best_top_left = (
                roi_x + local_top_left[0],
                roi_y + local_top_left[1],
            )
    return best_confidence, best_top_left


def add_label(
    frame,
    top_left: tuple[int, int],
    template_size: tuple[int, int],
    confidence: float,
    threshold: float,
    tenon_area: float | None,
    measurement_error: str | None,
) -> None:
    """Draw the best-match rectangle, confidence, and tenon measurement."""
    x, y = top_left
    template_width, template_height = template_size
    accepted = confidence >= threshold
    color = (0, 200, 0) if accepted else (0, 0, 255)
    status = "MATCH" if accepted else "LOW"
    labels = [f"{status}  confidence: {confidence:.3f}"]
    if tenon_area is not None:
        labels.append(f"tenon area: {tenon_area:.2f} px^2")
    elif accepted and measurement_error:
        labels.append("tenon area: unavailable")

    cv2.rectangle(
        frame,
        (x, y),
        (x + template_width, y + template_height),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    sizes = [cv2.getTextSize(text, font, font_scale, thickness) for text in labels]
    text_width = max(size[0][0] for size in sizes)
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    line_step = text_height + baseline + 5
    label_height = line_step * len(labels) + 5

    text_x = max(0, min(x, frame.shape[1] - text_width - 8))
    if y >= label_height + 4:
        box_top = y - label_height - 3
    else:
        box_top = min(frame.shape[0] - label_height - 1, y + 3)
    box_top = max(0, box_top)
    box_bottom = min(frame.shape[0] - 1, box_top + label_height)
    cv2.rectangle(
        frame,
        (text_x, box_top),
        (min(frame.shape[1] - 1, text_x + text_width + 8), box_bottom),
        color,
        -1,
    )
    for index, label in enumerate(labels):
        text_y = box_top + text_height + 3 + index * line_step
        cv2.putText(
            frame,
            label,
            (text_x + 4, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def process_video(args: argparse.Namespace) -> int:
    if not args.template.is_file():
        print(f"Error: template not found: {args.template}", file=sys.stderr)
        return 1
    if not args.video.is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1
    if args.output.resolve() == args.video.resolve():
        print("Error: output video must differ from input video", file=sys.stderr)
        return 1
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between 0 and 1", file=sys.stderr)
        return 1

    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        print(f"Error: could not read template: {args.template}", file=sys.stderr)
        return 1
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_height, template_width = template_gray.shape

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Error: could not open video: {args.video}", file=sys.stderr)
        return 1

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if template_width > frame_width or template_height > frame_height:
        capture.release()
        print(
            "Error: template is larger than the video frame "
            f"({template_width}x{template_height} vs {frame_width}x{frame_height})",
            file=sys.stderr,
        )
        return 1

    try:
        rois = load_rois(
            args.roi_file,
            (frame_width, frame_height),
            (template_width, template_height),
        )
    except ValueError as error:
        capture.release()
        print(f"Error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        video_fourcc(args.output),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        print(f"Error: could not create output video: {args.output}", file=sys.stderr)
        return 1

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    accepted_count = 0
    measured_count = 0
    measurement_failure_count = 0

    print(f"Template: {args.template} ({template_width}x{template_height})")
    print(f"Video:    {args.video} ({frame_width}x{frame_height}, {fps:.2f} fps)")
    print(f"Output:   {args.output}")
    print(f"ROI file: {args.roi_file} ({len(rois)} valid region(s))")
    print(f"Threshold: {args.threshold:.3f}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            confidence, top_left = find_best_roi_match(
                frame_gray, template_gray, rois
            )

            for roi_x, roi_y, roi_width, roi_height in rois:
                cv2.rectangle(
                    frame,
                    (roi_x, roi_y),
                    (roi_x + roi_width, roi_y + roi_height),
                    (255, 120, 0),
                    1,
                )

            tenon_area = None
            measurement_error = None
            if confidence >= args.threshold:
                accepted_count += 1
                match_x, match_y = top_left
                matched_crop = frame[
                    match_y : match_y + template_height,
                    match_x : match_x + template_width,
                ].copy()
                try:
                    tenon_polygon = detect_tenon_polygon(matched_crop)
                    tenon_area = float(cv2.contourArea(tenon_polygon))
                    measured_count += 1

                    full_frame_polygon = np.rint(tenon_polygon).astype(np.int32)
                    full_frame_polygon[:, 0] += match_x
                    full_frame_polygon[:, 1] += match_y
                    cv2.polylines(
                        frame,
                        [full_frame_polygon],
                        True,
                        (0, 215, 255),
                        2,
                        cv2.LINE_AA,
                    )
                except ValueError as error:
                    measurement_error = str(error)
                    measurement_failure_count += 1
            add_label(
                frame,
                top_left,
                (template_width, template_height),
                confidence,
                args.threshold,
                tenon_area,
                measurement_error,
            )
            writer.write(frame)
            frame_number += 1

            if args.display:
                cv2.imshow("Template matching", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    break

            if frame_number % 100 == 0:
                total_text = str(total_frames) if total_frames > 0 else "?"
                print(f"Processed {frame_number}/{total_text} frames", end="\r")
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if frame_number == 0:
        print("Error: no frames could be read from the video", file=sys.stderr)
        return 1

    print(
        f"\nDone: {frame_number} frames, {accepted_count} accepted matches "
        f"({accepted_count / frame_number:.1%})."
    )
    print(
        f"Tenon area measured on {measured_count} frame(s); "
        f"{measurement_failure_count} measurement failure(s)."
    )
    print(f"Saved annotated video to: {args.output}")
    print("Note: the OpenCV output contains video only; source audio is not copied.")
    return 0


def main() -> None:
    raise SystemExit(process_video(parse_args()))


if __name__ == "__main__":
    main()
