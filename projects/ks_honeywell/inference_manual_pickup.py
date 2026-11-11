"""Annotate all object/hand detections and manual pickup events in an MP4."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys


MANUALS = ("manual1", "manual2", "manual3", "manual4")


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


def draw_frame(cv2, frame, objects, hands, contacts, states, timestamp):
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
        label(f"{name} {confidence:.2f}{suffix}", x1, y1 - 6, color)

    label(f"Video time: {timestamp:.2f}s | Pickup = hand overlap", 8,
          line_height, (255, 255, 255))
    for index, name in enumerate(MANUALS, 2):
        state = states[name]
        status = "CONTACT" if name in contacts else "NO CONTACT"
        label(f"{name}: {state.count} pickups | {status}", 8,
              index * line_height, (255, 255, 255))
    recent = [name for name in MANUALS if states[name].last_event is not None
              and timestamp - states[name].last_event < 2.0]
    if recent:
        label("PICKUP: " + ", ".join(recent), 8, 6 * line_height, (0, 255, 255))
    return frame


def validate_paths(args):
    for label, value in (("Source video", args.source),
                         ("Manual weights", args.manual_weights),
                         ("Hand weights", args.hand_weights)):
        if not Path(value).is_file():
            raise ValueError(f"{label} not found: {value}")
    output = Path(args.output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if output.resolve() in {Path(value).resolve() for value in
                            (args.source, args.manual_weights, args.hand_weights)}:
        raise ValueError("Output must not overwrite the source video or model weights")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new output path: {output}")


def run(args, *, cv2_module=None, model_factory=None):
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
    states = {name: PickupState() for name in MANUALS}
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
        manual_model = model_factory(str(args.manual_weights))
        hand_model = model_factory(str(args.hand_weights))
        for label, model in (("Manual", manual_model), ("Hand", hand_model)):
            if model.task != "detect":
                raise ValueError(f"{label} checkpoint must be an object detection model")
        manual_class_ids(manual_model.names)
        names = manual_model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_names = {int(key): str(value) for key, value in items}
        height, width = frame.shape[:2]
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
            hands = extract_detections(hand_model.predict(
                frame, conf=args.hand_conf, **options)[0])
            timestamp = frame_index / fps
            contacts = contact_classes(objects, hands)
            for name, state in states.items():
                if state.update(name in contacts, timestamp,
                                args.contact_duration, args.release_duration):
                    print(f"{timestamp:.3f}s PICKUP {name} count={state.count}")
            writer.write(draw_frame(cv2, frame, objects, hands, contacts, states, timestamp))
            frame_index += 1
            if frame_index % 300 == 0:
                print(f"Processed {frame_index} frames ({frame_index / fps:.1f}s)")
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
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
    parser.add_argument("--output", required=True, help="New annotated .mp4 file")
    parser.add_argument("--device", default=None, help="Ultralytics device: 0, cpu, mps; default auto")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--manual-conf", type=confidence, default=0.25,
                        help="Confidence threshold for all classes in the custom object model")
    parser.add_argument("--hand-conf", type=confidence, default=0.25)
    parser.add_argument("--contact-duration", type=nonnegative_float, default=0.2,
                        help="Continuous overlap seconds before pickup")
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
