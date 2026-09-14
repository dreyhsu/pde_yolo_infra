#!/usr/bin/env python3
"""Measure the visible MCIO-card tenon area with OpenCV.

This detector is intended for tightly cropped MCIO-card images with the same
viewpoint as ``nk_apl_mcio/images/template.png`` and ``ng.png``.  It finds the
outer horizontal edges and the center notch from image gradients, constructs a
tenon polygon, and reports its area in square pixels.

Example:
    conda run -n tf python transfer/measure_tenon_area.py \
        transfer/nk_apl_mcio/images/template.png \
        transfer/nk_apl_mcio/images/ng.png
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the visible tenon area in cropped MCIO-card images."
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="One or more tightly cropped MCIO-card images",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "Optional CVAT annotations.xml used only to compare predictions with "
            "the tenon ground truth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tenon_area_results",
        help="Directory for masks and overlays (default: transfer/tenon_area_results)",
    )
    return parser.parse_args()


def range_bounds(length: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    """Convert a normalized interval to a safe, non-empty integer interval."""
    start = max(0, min(length - 1, round(start_ratio * length)))
    end = max(start + 1, min(length, round(end_ratio * length)))
    return start, end


def strongest_x(
    profile: np.ndarray,
    start: int,
    end: int,
    find_maximum: bool,
) -> int:
    section = profile[start:end]
    offset = int(np.argmax(section) if find_maximum else np.argmin(section))
    return start + offset


def fit_edge_line(points: list[tuple[float, float]]):
    """Robustly fit y = f(x) through edge candidates using OpenCV."""
    array = np.asarray(points, dtype=np.float32)
    median_y = float(np.median(array[:, 1]))
    filtered = array[np.abs(array[:, 1] - median_y) <= 2.0]
    if len(filtered) < 2:
        filtered = array

    vx, vy, line_x, line_y = cv2.fitLine(
        filtered, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).reshape(-1)
    if abs(float(vx)) < 1e-6:
        raise ValueError("Could not fit a horizontal tenon edge")

    def y_at(x: float) -> float:
        return float(line_y + (x - line_x) * vy / vx)

    return y_at


def collect_vertical_edge_candidates(
    gradient_y: np.ndarray,
    x_values: list[int],
    y_start: int,
    y_end: int,
    find_maximum: bool,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x in x_values:
        profile = gradient_y[y_start:y_end, x]
        offset = int(np.argmax(profile) if find_maximum else np.argmin(profile))
        points.append((float(x), float(y_start + offset)))
    return points


def detect_tenon_polygon(image: np.ndarray) -> np.ndarray:
    """Return the detected eight-point tenon polygon as float32 coordinates."""
    if image is None or image.ndim != 3:
        raise ValueError("Expected a readable BGR color image")

    height, width = image.shape[:2]
    if width < 50 or height < 40:
        raise ValueError(
            f"Image is too small for tenon detection: {width}x{height}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    # Locate the two outer tenon ends using vertical edge polarity.
    edge_y0, edge_y1 = range_bounds(height, 0.45, 0.65)
    outer_profile = gradient_x[edge_y0:edge_y1].mean(axis=0)
    left0, left1 = range_bounds(width, 0.08, 0.20)
    right0, right1 = range_bounds(width, 0.80, 0.96)
    left_x = strongest_x(outer_profile, left0, left1, find_maximum=True)
    right_x = strongest_x(outer_profile, right0, right1, find_maximum=False)

    # Locate the left and right edges of the central notch.
    notch_y0, notch_y1 = range_bounds(height, 0.38, 0.55)
    notch_profile = gradient_x[notch_y0:notch_y1].mean(axis=0)
    notch_left0, notch_left1 = range_bounds(width, 0.32, 0.50)
    notch_right0, notch_right1 = range_bounds(width, 0.50, 0.70)
    notch_left_x = strongest_x(
        notch_profile, notch_left0, notch_left1, find_maximum=True
    )
    notch_right_x = strongest_x(
        notch_profile, notch_right0, notch_right1, find_maximum=False
    )

    if not left_x < notch_left_x < notch_right_x < right_x:
        raise ValueError(
            "Tenon edges were not found in the expected order; check the crop and viewpoint"
        )

    # Fit the slightly sloped upper edge while excluding the center notch.
    top_y0, top_y1 = range_bounds(height, 0.35, 0.55)
    upper_x_values = list(range(left_x + 2, notch_left_x - 1))
    upper_x_values += list(range(notch_right_x + 2, right_x - 1))
    upper_candidates = collect_vertical_edge_candidates(
        gradient_y,
        upper_x_values,
        top_y0,
        top_y1,
        find_maximum=False,
    )
    upper_line = fit_edge_line(upper_candidates)

    # The strongest Sobel response is centered on the transition. Half a pixel
    # moves the polygon onto the physical lower side of that upper edge.
    upper_edge_offset = 0.5

    # Find the darkest row beneath the central tab; this is the notch boundary.
    center_y0, center_y1 = range_bounds(height, 0.35, 0.62)
    center_x0 = notch_left_x + 2
    center_x1 = notch_right_x - 1
    center_profile = gray[center_y0:center_y1, center_x0:center_x1].mean(axis=1)
    notch_y = float(center_y0 + int(np.argmin(center_profile)))

    # Fit the lower tenon edge across its full width.
    bottom_y0, bottom_y1 = range_bounds(height, 0.55, 0.78)
    lower_x_values = list(range(left_x + 2, right_x - 1))
    lower_candidates = collect_vertical_edge_candidates(
        gradient_y,
        lower_x_values,
        bottom_y0,
        bottom_y1,
        find_maximum=True,
    )
    lower_line = fit_edge_line(lower_candidates)

    polygon = np.asarray(
        [
            (left_x, upper_line(left_x) + upper_edge_offset),
            (notch_left_x, upper_line(notch_left_x) + upper_edge_offset),
            (notch_left_x, notch_y),
            (notch_right_x, notch_y),
            (notch_right_x, upper_line(notch_right_x) + upper_edge_offset),
            (right_x, upper_line(right_x) + upper_edge_offset),
            (right_x, lower_line(right_x)),
            (left_x, lower_line(left_x)),
        ],
        dtype=np.float32,
    )

    if cv2.contourArea(polygon) <= 0:
        raise ValueError("Detected tenon polygon has an invalid area")
    return polygon


def read_ground_truth(annotation_path: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Load CVAT tenon polygons for validation; they do not affect detection."""
    root = ET.parse(annotation_path).getroot()
    ground_truth: dict[str, tuple[np.ndarray, float]] = {}
    for image_element in root.findall("image"):
        polygon_element = image_element.find("polygon[@label='tenon']")
        if polygon_element is None:
            continue
        points = [
            tuple(float(value) for value in pair.split(","))
            for pair in polygon_element.attrib["points"].split(";")
        ]
        polygon = np.asarray(points, dtype=np.float32)
        ground_truth[Path(image_element.attrib["name"]).name] = (
            polygon,
            float(cv2.contourArea(polygon)),
        )
    return ground_truth


def find_default_annotations(images: list[Path]) -> Path | None:
    """Find annotations.xml one directory above a shared images directory."""
    candidates = {image.resolve().parent.parent / "annotations.xml" for image in images}
    if len(candidates) == 1:
        candidate = candidates.pop()
        if candidate.is_file():
            return candidate
    return None


def save_visualizations(
    image: np.ndarray,
    polygon: np.ndarray,
    area: float,
    output_dir: Path,
    stem: str,
    ground_truth_polygon: np.ndarray | None,
) -> tuple[Path, Path]:
    integer_polygon = np.rint(polygon).astype(np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [integer_polygon], 255)

    overlay = image.copy()
    tint = image.copy()
    cv2.fillPoly(tint, [integer_polygon], (0, 215, 255))
    overlay = cv2.addWeighted(tint, 0.30, overlay, 0.70, 0)
    cv2.polylines(overlay, [integer_polygon], True, (0, 215, 255), 1, cv2.LINE_AA)

    if ground_truth_polygon is not None:
        gt_integer = np.rint(ground_truth_polygon).astype(np.int32)
        cv2.polylines(overlay, [gt_integer], True, (255, 0, 255), 1, cv2.LINE_AA)

    # Upscale the tiny source crops so the boundary is easy to inspect.
    scale = max(1, round(600 / image.shape[1]))
    overlay = cv2.resize(
        overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    mask_large = cv2.resize(
        mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        overlay,
        f"Detected area: {area:.2f} px^2",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    overlay_path = output_dir / f"{stem}_tenon_overlay.png"
    mask_path = output_dir / f"{stem}_tenon_mask.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not write {overlay_path}")
    if not cv2.imwrite(str(mask_path), mask_large):
        raise OSError(f"Could not write {mask_path}")
    return overlay_path, mask_path


def main() -> int:
    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        print(f"Error: image not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    annotation_path = args.annotations or find_default_annotations(args.images)
    ground_truth: dict[str, tuple[np.ndarray, float]] = {}
    if annotation_path is not None:
        if not annotation_path.is_file():
            print(f"Error: annotations not found: {annotation_path}", file=sys.stderr)
            return 1
        ground_truth = read_ground_truth(annotation_path)
        print(f"Ground truth: {annotation_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\nTenon area results (px^2)")
    print("-" * 75)
    print(f"{'image':<20} {'detected':>12} {'ground truth':>14} {'error':>12}")
    print("-" * 75)

    failed = False
    for image_path in args.images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        try:
            polygon = detect_tenon_polygon(image)
            detected_area = float(cv2.contourArea(polygon))
            gt_entry = ground_truth.get(image_path.name)
            gt_polygon = gt_entry[0] if gt_entry else None

            if gt_entry:
                gt_area = gt_entry[1]
                error_percent = 100.0 * (detected_area - gt_area) / gt_area
                gt_text = f"{gt_area:.2f}"
                error_text = f"{error_percent:+.2f}%"
            else:
                gt_text = "n/a"
                error_text = "n/a"

            print(
                f"{image_path.name:<20} {detected_area:>12.2f} "
                f"{gt_text:>14} {error_text:>12}"
            )
            save_visualizations(
                image,
                polygon,
                detected_area,
                args.output_dir,
                image_path.stem,
                gt_polygon,
            )
        except (ValueError, OSError) as error:
            failed = True
            print(f"{image_path.name:<20} ERROR: {error}", file=sys.stderr)

    print("-" * 75)
    print(f"Visualizations: {args.output_dir}")
    if ground_truth:
        print("Overlay colors: yellow = detected, magenta = ground truth")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
