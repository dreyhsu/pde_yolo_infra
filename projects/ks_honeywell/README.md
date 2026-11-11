# Honeywell manual pickup detection

Runs a custom **YOLO26m** detector for `manual1`, `manual2`, `manual3`, and
`manual4`, plus a **YOLOv8n hand detector**, in the same Ultralytics environment.
The result is an annotated MP4 with manual/hand boxes, confidence, live contact
status, pickup counts, and a two-second pickup banner. Events and final counts
are also printed to the terminal. Audio is not copied.

## Setup

Use Python 3.10–3.12 in an environment with a suitable PyTorch installation for
your device. From the repository root:

```sh
python -m pip install -r projects/ks_honeywell/requirements.txt
```

If you already have a working YOLO26 environment, install these dependencies
there; a separate YOLOv8 environment is unnecessary.

Download the public hand checkpoint from its publisher,
[Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer):

```sh
mkdir -p projects/ks_honeywell/weights
curl -fL --retry 3 -o projects/ks_honeywell/weights/hand_yolov8n.pt https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt
```

On Windows, create the folder with File Explorer and download
[hand_yolov8n.pt](https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt)
into it, or use `curl.exe` with the command above after creating the folder.
This model detects whole hands, not palm landmarks. Inference does not download
weights automatically; both checkpoints must exist locally.

## Run

On Windows, edit `SOURCE`, `MANUAL_WEIGHTS`, `HAND_WEIGHTS`, and `OUTPUT` in
[`run_inference_manual_pickup.bat`](run_inference_manual_pickup.bat), then run it
from Anaconda Prompt or double-click it if Conda is on your PATH. It activates
the Conda environment `5090` and uses GPU `0`. Relative paths are resolved from
the repository root, regardless of the launcher's working directory. Install the
dependencies and download the hand weights using the setup instructions above
first. Choose a new `OUTPUT` path if a previous output already exists.

Supply your **trained YOLO26m** checkpoint, not the standard COCO `yolo26m.pt`.
Its class names must include the exact names `manual1` through `manual4`; class
IDs may be in any order and other classes are ignored.

```sh
python projects/ks_honeywell/inference_manual_pickup.py --source "video/input.mp4" --manual-weights "logs/ks_honeywell/yolo26m/weights/best.pt" --hand-weights "projects/ks_honeywell/weights/hand_yolov8n.pt" --output "projects/ks_honeywell/output/pickups.mp4" --device 0
```

Use `--device cpu` for CPU, `--device mps` for Apple Silicon, or omit it for
Ultralytics' automatic device selection. Use a new output filename on each run;
the script refuses to overwrite an existing file. Output folders are created.

Options: `--imgsz 640`, `--manual-conf 0.25`, `--hand-conf 0.25`,
`--contact-duration 0.2`, and `--release-duration 0.3`.
Run with `--help` for all arguments. Set contact duration to `0` for immediate
pickup on overlap. Increase image size if hands are small (at a speed cost).

## Pickup rule and timing

- Any positive-area intersection between a hand box and a manual box is contact;
  boxes that only touch at an edge do not qualify.
- A pickup fires after 0.2 seconds of uninterrupted detected contact. Continuous
  contact produces one event. A 0.3-second interval without contact rearms that
  manual class. Missing hand/manual detections count as no contact.
- State and counts are per manual **class**, not physical object or operator.
  Any box of a class can maintain its contact; simultaneous contact with different
  manual classes can trigger each class independently.
- Time is decoded frame index / source FPS, starting at zero. Thresholds are
  evaluated at frame timestamps, so the first contacting frame starts the timer
  and thresholds are rounded up to the next frame. Inference speed has no effect.
- Output preserves decoded dimensions and reported FPS using the `mp4v` codec.
  Variable-frame-rate inputs are written at constant reported FPS; normalize them
  beforehand if exact source timestamps matter. Invalid FPS is rejected.

“Pickup” is the requested overlap heuristic: it does not prove physical contact,
lifting, or manual movement. Occlusion can interrupt contact detection; tune
confidence and durations against your camera footage. This public hand model
has not been validated on this production line.

If processing fails after output creation, a partial MP4 may remain; the script
reports an error and releases video resources. Use a new path or remove the
partial file before retrying.

## Validation

The lightweight tests need only Python and exercise timing, overlap, class
validation, and video processing/error handling using test adapters:

```sh
python -m unittest discover -s projects/ks_honeywell -p "test_*.py" -v
```

For a real-model smoke test, run on a short representative video with your manual
weights, play the output, and check both kinds of boxes, contact timing, counts,
duration, FPS, and dimensions. No trained manual checkpoint or sample footage is
bundled with this project.
