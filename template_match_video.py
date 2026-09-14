#!/usr/bin/env python3
"""Run simple OpenCV template matching on every frame of a video.

The output video contains the best-match bounding box and confidence score on
every frame. Matches at or above the chosen threshold are drawn in green;
lower-confidence best candidates are drawn in red.

Example:
    python template_match_video.py template.png input.mp4 output.mp4
    python template_match_video.py template.png input.mp4 output.mp4 --threshold 0.80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print(
        "OpenCV is not installed in this Python environment. "
        "Install it with: python3 -m pip install opencv-python",
        file=sys.stderr,
    )
    raise SystemExit(1)


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


def add_label(
    frame,
    top_left: tuple[int, int],
    template_size: tuple[int, int],
    confidence: float,
    threshold: float,
) -> None:
    """Draw the best-match rectangle and a readable confidence label."""
    x, y = top_left
    template_width, template_height = template_size
    accepted = confidence >= threshold
    color = (0, 200, 0) if accepted else (0, 0, 255)
    status = "MATCH" if accepted else "LOW"
    label = f"{status}  confidence: {confidence:.3f}"

    cv2.rectangle(
        frame,
        (x, y),
        (x + template_width, y + template_height),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    # Put the label above the box when possible, otherwise put it just inside.
    text_x = max(0, min(x, frame.shape[1] - text_width - 8))
    text_y = y - 8 if y - text_height - baseline - 8 >= 0 else y + text_height + 8
    box_top = max(0, text_y - text_height - 5)
    box_bottom = min(frame.shape[0] - 1, text_y + baseline + 4)
    cv2.rectangle(
        frame,
        (text_x, box_top),
        (min(frame.shape[1] - 1, text_x + text_width + 8), box_bottom),
        color,
        -1,
    )
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

    print(f"Template: {args.template} ({template_width}x{template_height})")
    print(f"Video:    {args.video} ({frame_width}x{frame_height}, {fps:.2f} fps)")
    print(f"Output:   {args.output}")
    print(f"Threshold: {args.threshold:.3f}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score_map = cv2.matchTemplate(
                frame_gray, template_gray, cv2.TM_CCOEFF_NORMED
            )
            _, confidence, _, top_left = cv2.minMaxLoc(score_map)

            if confidence >= args.threshold:
                accepted_count += 1
            add_label(
                frame,
                top_left,
                (template_width, template_height),
                confidence,
                args.threshold,
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
    print(f"Saved annotated video to: {args.output}")
    print("Note: the OpenCV output contains video only; source audio is not copied.")
    return 0


def main() -> None:
    raise SystemExit(process_video(parse_args()))


if __name__ == "__main__":
    main()
