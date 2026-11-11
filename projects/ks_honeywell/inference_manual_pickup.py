"""Annotate all object/hand detections and manual pickup events in an MP4."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys


MANUALS = ("manual1", "manual2", "manual3", "manual4")
INDEX_TIP = 8


def hand_contact_quadrant(box, handedness):
    """Use displayed-image coordinates; the input video is already mirrored."""
    x1, y1, x2, y2 = box
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    if handedness == "right":
        return (mid_x, mid_y, x2, y2)
    if handedness == "left":
        return (x1, mid_y, mid_x, y2)
    raise ValueError(f"Unknown handedness: {handedness}")


class HandSideMemory:
    """Refresh references on two-hand frames; match lone hands to that pair."""

    def __init__(self, image_left_hand="right"):
        self.image_left_hand = image_left_hand
        self.centers = {}

    @staticmethod
    def center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def update(self, hands):
        valid = [(name, conf, box) for name, conf, box in hands
                 if all(math.isfinite(value) for value in box)
                 and box[2] > box[0] and box[3] > box[1]]
        # One operator: use the two highest-confidence hands if extras appear.
        pair = sorted(valid, key=lambda item: -item[1])[:2]
        if len(pair) == 2:
            pair.sort(key=lambda item: self.center(item[2]))
            other = "left" if self.image_left_hand == "right" else "right"
            assigned = [(side, conf, box) for side, (_, conf, box)
                        in zip((self.image_left_hand, other), pair)]
            self.centers = {side: self.center(box) for side, _, box in assigned}
            return assigned
        if not pair or not self.centers:
            return []
        _, conf, box = pair[0]
        center = self.center(box)
        # Dict order breaks exact-distance ties toward the saved image-left hand.
        side = min(self.centers, key=lambda key: math.dist(center, self.centers[key]))
        return [(side, conf, box)]


def validate_roi(box, width, height):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(type(value) is int for value in box)):
        raise ValueError("ROI must contain four integer xyxy coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("ROI must have positive area and lie inside the source frame")
    return tuple(box)


def load_roi(path, width, height):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read ROI file {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("ROI file must contain an object")
    if data.get("frame_width") != width or data.get("frame_height") != height:
        raise ValueError("ROI frame dimensions do not match this video; select the ROI again")
    return validate_roi(data.get("roi_xyxy"), width, height)


def save_roi(path, box, width, height):
    box = validate_roi(box, width, height)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {"frame_width": width, "frame_height": height, "roi_xyxy": list(box)}
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def inside_roi(box, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    return (roi[0] <= x1 < x2 <= roi[2] and roi[1] <= y1 < y2 <= roi[3])


def detections_in_roi(detections, roi):
    return [item for item in detections if inside_roi(item[2], roi)]


class ManualBoxMemory:
    """Retain each manual class's latest detections through occlusion."""

    def __init__(self):
        self.boxes = {}

    def update(self, objects):
        visible = {name for name, _, _ in objects if name in MANUALS}
        for name in visible:
            self.boxes[name] = [item for item in objects if item[0] == name]
        remembered = set(self.boxes) - visible
        cached = [item for name in MANUALS if name in remembered for item in self.boxes[name]]
        return objects + cached, remembered


def fingertip_contacts(objects, fingertips):
    """Only the index tip, including the manual box boundary, qualifies."""
    return {name for name, _, (x1, y1, x2, y2) in objects if name in MANUALS
            and x2 > x1 and y2 > y1
            and any(math.isfinite(x) and math.isfinite(y)
                    and x1 <= x <= x2 and y1 <= y <= y2 for x, y in fingertips)}


def pixel_landmarks(result, width, height):
    hands = []
    for hand in result.hand_landmarks:
        if len(hand) != 21:
            continue
        points = [(point.x * width, point.y * height) for point in hand]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            continue
        x, y = points[INDEX_TIP]
        if 0 <= x < width and 0 <= y < height:
            hands.append(points)
    return hands


class IndexFingerLandmarker:
    """MediaPipe Tasks video inference; CPU landmarks alongside GPU YOLO."""

    def __init__(self, model_path, num_hands, cv2_module):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.cv2 = cv2_module
        self.last_timestamp_ms = -1
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.model = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp):
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe requires strictly increasing integer milliseconds.
        timestamp_ms = max(self.last_timestamp_ms + 1, round(timestamp * 1000))
        result = self.model.detect_for_video(image, timestamp_ms)
        self.last_timestamp_ms = timestamp_ms
        height, width = frame.shape[:2]
        return pixel_landmarks(result, width, height)

    def close(self):
        self.model.close()


def manual_class_ids(names):
    """Resolve names rather than assuming checkpoint class order."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = {int(key): str(value) for key, value in items if value in MANUALS}
    missing = set(MANUALS) - set(ids.values())
    if missing:
        raise ValueError(f"Manual checkpoint missing classes: {', '.join(sorted(missing))}")
    return ids


def intersects(a, b):
    """Edge-only touching has zero area and is not contact."""
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def contact_classes(manuals, hands):
    # Detections are (name, confidence, xyxy). Multiple instances share class state.
    return {name for name, _, box in manuals
            if name in MANUALS
            and any(intersects(box, hand_box) for _, _, hand_box in hands)}


@dataclass
class PickupState:
    count: int = 0
    armed: bool = True
    contact_since: float | None = None
    absent_since: float | None = None
    last_event: float | None = None

    def update(self, contact, timestamp, contact_duration, release_duration):
        """Use source-video seconds, independent of inference processing speed."""
        if not contact:
            self.contact_since = None
            if self.absent_since is None:
                self.absent_since = timestamp
            if timestamp - self.absent_since + 1e-9 >= release_duration:
                self.armed = True
            return False
        self.absent_since = None
        if self.contact_since is None:
            self.contact_since = timestamp
        if self.armed and timestamp - self.contact_since + 1e-9 >= contact_duration:
            self.count += 1
            self.armed = False
            self.last_event = timestamp
            return True
        return False


def extract_detections(result, class_names=None):
    if result.boxes is None:
        return []
    detections = []
    for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
        class_id = int(class_id)
        if class_names is not None and class_id not in class_names:
            continue
        name = class_names[class_id] if class_names is not None else "hand"
        detections.append((name, confidence, (x1, y1, x2, y2)))
    return detections


def draw_frame(cv2, frame, objects, hands, contacts, states, timestamp,
               landmarks=(), contact_mode="hand-box", remembered=(), contact_regions=(),
               roi=None):
    height, width = frame.shape[:2]
    scale = max(0.35, min(0.7, width / 1400))
    line_height = max(17, round(30 * scale / 0.7))

    def label(text, x, y, color):
        x, y = max(0, min(x, width - 1)), max(12, min(y, height - 2))
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    for name, confidence, box in objects + hands:
        color = ((255, 200, 0) if name == "hand" else
                 (0, 165, 255) if name in contacts else (0, 220, 0))
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(frame, (max(0, x1), max(0, y1)),
                      (min(width - 1, x2), min(height - 1, y2)), color, 2)
        suffix = " CONTACT" if name in contacts else ""
        if name in remembered:
            suffix += " REMEMBERED"
        label(f"{name} {confidence:.2f}{suffix}", x1, y1 - 6, color)

    if contact_mode == "hand-quadrant":
        for _, _, (x1, y1, x2, y2) in hands:
            mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.line(frame, (mid_x, int(y1)), (mid_x, int(y2)), (255, 200, 0), 1)
            cv2.line(frame, (int(x1), mid_y), (int(x2), mid_y), (255, 200, 0), 1)
        for side, _, box in contact_regions:
            touching = (inside_roi(box, roi) and bool(contact_classes(
                detections_in_roi(objects, roi), [(side, 1.0, box)])))
            color = (0, 0, 255) if touching else (255, 0, 255)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label(f"{side.upper()} HAND ZONE", x1, y2 - 6, color)

    for hand in landmarks:
        for start, end in ((5, 6), (6, 7), (7, INDEX_TIP)):
            a, b = hand[start], hand[end]
            cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 2)
        tip = hand[INDEX_TIP]
        touching = bool(fingertip_contacts(detections_in_roi(objects, roi), [tip]))
        color = (0, 0, 255) if touching else (255, 0, 255)
        cv2.circle(frame, tuple(map(int, tip)), 6, color, -1)
        label("INDEX TIP", int(tip[0]) + 8, int(tip[1]) - 8, color)

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (min(x2, width - 1), min(y2, height - 1)),
                      (0, 255, 255), 2)
        label("COUNTING ROI", x1 + 4, y1 + 18, (0, 255, 255))

    touch_mode = contact_mode != "hand-box"
    event = "TOUCH" if touch_mode else "PICKUP"
    rule = {"index-finger": "Touch = index tip in manual",
            "hand-quadrant": "Touch = lower hand quarter overlap",
            "hand-box": "Pickup = hand overlap"}[contact_mode]
    label(f"Video time: {timestamp:.2f}s | {rule}", 8,
          line_height, (255, 255, 255))
    for index, name in enumerate(MANUALS, 2):
        state = states[name]
        status = "CONTACT" if name in contacts else "NO CONTACT"
        unit = "touches" if touch_mode else "pickups"
        label(f"{name}: {state.count} {unit} | {status}", 8,
              index * line_height, (255, 255, 255))
    recent = [name for name in MANUALS if states[name].last_event is not None
              and timestamp - states[name].last_event < 2.0]
    if recent:
        label(event + ": " + ", ".join(recent), 8, 6 * line_height, (0, 255, 255))
    return frame


def validate_paths(args):
    for label, value in (("Source video", args.source),
                         ("Manual weights", args.manual_weights),
                         ("Hand weights", args.hand_weights)):
        if not Path(value).is_file():
            raise ValueError(f"{label} not found: {value}")
    if args.contact_mode == "index-finger" and not Path(args.landmark_model).is_file():
        raise ValueError(f"Hand landmark model not found: {args.landmark_model}; "
                         "download hand_landmarker.task as described in README.md")
    output = Path(args.output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if output.resolve() in {Path(value).resolve() for value in
                            (args.source, args.manual_weights, args.hand_weights,
                             args.landmark_model)}:
        raise ValueError("Output must not overwrite the source video or model weights")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new output path: {output}")


def run(args, *, cv2_module=None, model_factory=None, landmark_factory=None):
    """Process one frame at a time; optional adapters support lightweight tests."""
    validate_paths(args)
    if cv2_module is None:
        import cv2 as cv2_module
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    cv2 = cv2_module
    cap = cv2.VideoCapture(str(args.source))
    writer = None
    landmarker = None
    states = {name: PickupState() for name in MANUALS}
    memory = ManualBoxMemory()
    hand_memory = HandSideMemory(args.image_left_hand)
    frame_index = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {args.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source FPS: {fps}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Source video contains no readable frames")
        height, width = frame.shape[:2]
        roi = load_roi(args.roi, width, height) if args.roi else None
        manual_model = model_factory(str(args.manual_weights))
        hand_model = model_factory(str(args.hand_weights))
        for label, model in (("Manual", manual_model), ("Hand", hand_model)):
            if model.task != "detect":
                raise ValueError(f"{label} checkpoint must be an object detection model")
        manual_class_ids(manual_model.names)
        names = manual_model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_names = {int(key): str(value) for key, value in items}
        if args.contact_mode == "index-finger":
            factory = landmark_factory or IndexFingerLandmarker
            landmarker = factory(args.landmark_model, args.num_hands, cv2)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot create MP4 writer: {args.output}")
        options = dict(imgsz=args.imgsz, verbose=False)
        if args.device is not None:
            options["device"] = args.device
        while ok and frame is not None:
            if frame.shape[:2] != (height, width):
                raise ValueError("Source frame dimensions changed during decoding")
            # Both detectors see the unannotated original frame.
            objects = extract_detections(manual_model.predict(
                frame, conf=args.manual_conf, **options)[0], class_names)
            objects, remembered = memory.update(objects)
            hands = extract_detections(hand_model.predict(
                frame, conf=args.hand_conf, **options)[0])
            timestamp = frame_index / fps
            landmarks = landmarker.detect(frame, timestamp) if landmarker is not None else []
            count_objects = detections_in_roi(objects, roi)
            contact_regions = []
            if args.contact_mode == "index-finger":
                contacts = fingertip_contacts(count_objects, [hand[INDEX_TIP] for hand in landmarks])
            elif args.contact_mode == "hand-quadrant":
                assigned = hand_memory.update(hands)
                contact_regions = [(side, conf, hand_contact_quadrant(box, side))
                                   for side, conf, box in assigned if inside_roi(box, roi)]
                contacts = contact_classes(count_objects, contact_regions)
            else:
                contacts = contact_classes(count_objects, detections_in_roi(hands, roi))
            for name, state in states.items():
                if state.update(name in contacts, timestamp,
                                args.contact_duration, args.release_duration):
                    event = "PICKUP" if args.contact_mode == "hand-box" else "TOUCH"
                    print(f"{timestamp:.3f}s {event} {name} count={state.count}")
            writer.write(draw_frame(cv2, frame, objects, hands, contacts, states, timestamp,
                                    landmarks, args.contact_mode, remembered, contact_regions, roi))
            frame_index += 1
            if frame_index % 300 == 0:
                print(f"Processed {frame_index} frames ({frame_index / fps:.1f}s)")
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if landmarker is not None:
            landmarker.close()
    print(f"Saved {frame_index} frames to {args.output}")
    print(" | ".join(f"{name}: {state.count}" for name, state in states.items()))
    return states


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def confidence(value):
    parsed = nonnegative_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--manual-weights", required=True, help="Custom YOLO26m best.pt")
    parser.add_argument("--hand-weights", default=str(
        Path(__file__).resolve().parent / "weights" / "hand_yolov8n.pt"))
    parser.add_argument("--contact-mode", choices=("hand-quadrant", "index-finger", "hand-box"),
                        default="hand-quadrant", help="Default: lower hand quarter overlaps a manual")
    parser.add_argument("--image-left-hand", choices=("left", "right"), default="right",
                        help="Image-left hand in two-hand frames (default: right, mirrored video)")
    parser.add_argument("--roi", help="Saved ROI JSON; only fully contained boxes count")
    parser.add_argument("--landmark-model", default=str(
        Path(__file__).resolve().parent / "weights" / "hand_landmarker.task"))
    parser.add_argument("--num-hands", type=positive_int, default=2,
                        help="Maximum hands for landmark inference (default: 2)")
    parser.add_argument("--output", required=True, help="New annotated .mp4 file")
    parser.add_argument("--device", default=None, help="Ultralytics device: 0, cpu, mps; default auto")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--manual-conf", type=confidence, default=0.25,
                        help="Confidence threshold for all classes in the custom object model")
    parser.add_argument("--hand-conf", type=confidence, default=0.25)
    parser.add_argument("--contact-duration", type=nonnegative_float, default=0.2,
                        help="Continuous contact seconds before a touch/pickup event")
    parser.add_argument("--release-duration", type=nonnegative_float, default=0.3,
                        help="Continuous absence seconds before rearming")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
