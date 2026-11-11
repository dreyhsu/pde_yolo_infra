# Honeywell manual contact detection

Runs your custom **YOLO26m** detector and a **YOLOv8n hand detector** together.
The default contact rule divides each hand box into four equal quarters and uses
the lower-right quarter of a right hand or the lower-left quarter of a left hand.
The input video is already mirrored; the script does not flip it again.

The annotated MP4 shows every detected custom-model class, hand boxes split into
quarters, the selected contact zones, remembered manual boxes, contact status,
running touch counts, and a two-second TOUCH banner. Audio is not copied.

## Setup

Use your existing Conda `5090` environment (Python 3.10–3.12 recommended) with
PyTorch installed for your GPU. Run from the repository root in Anaconda Prompt:

```bat
conda activate 5090
python -m pip install -r projects/ks_honeywell/requirements.txt
if not exist projects\ks_honeywell\weights mkdir projects\ks_honeywell\weights
curl.exe -fL --retry 3 -o projects\ks_honeywell\weights\hand_yolov8n.pt https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt
```

The public hand detector is published at
[Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer). The default mode needs
only your manual checkpoint and this hand checkpoint. MediaPipe and its `.task`
model are **not required**. Inference does not download weights automatically.

## Run and hand assignment

Edit `SOURCE`, `MANUAL_WEIGHTS`, `HAND_WEIGHTS`, `OUTPUT`, and `IMAGE_LEFT_HAND`
in [`run_inference_manual_pickup.bat`](run_inference_manual_pickup.bat). Run it
from Anaconda Prompt, or double-click it if Conda is on your PATH. The launcher
activates `5090`, uses GPU `0`, and resolves relative paths from the repository
root. It opens the ROI selector if the configured `ROI` file does not exist,
then reuses that saved ROI on later runs. Choose a new output filename for every run.

```sh
python projects/ks_honeywell/inference_manual_pickup.py --source "video/input.mp4" --manual-weights "logs/ks_honeywell/yolo26m/weights/best.pt" --hand-weights "projects/ks_honeywell/weights/hand_yolov8n.pt" --output "projects/ks_honeywell/output/pickups.mp4" --device 0 --contact-mode hand-quadrant --roi "projects/ks_honeywell/roi.json"
```

Supply a YOLO26m checkpoint trained on your custom classes, not standard COCO
weights. The exact names `manual1`–`manual4` must exist; IDs may be in any order.
All 11 classes are displayed, including `scanner_with_label`,
`scanner_without_label`, `scanner_in_bag`, `box_folded`, `scanner_in_box`,
`battery_in_bag`, and `label_on_box`. Only manuals participate in contact counting.

The public YOLO hand model has one class, `hand`: it does **not** predict
anatomical left/right handedness. The script remembers the two hand positions:

| Box in a two-hand frame | Assigned hand | Active quarter |
| --- | --- | --- |
| Image-left box | Right hand | Lower-right |
| Image-right box | Left hand | Lower-left |

Every frame with two valid detections refreshes both reference centers, sorted
left-to-right regardless of their position relative to the image midpoint. When
only one is detected, its center is matched to the closest saved center using
Euclidean distance. Single-hand frames do not move those reference positions.
An exact-distance tie uses the saved image-left hand. With zero detections,
no hand contact is generated. Before the first two-hand frame, single hands stay
unassigned and cannot trigger quadrant contact. If more than two are detected,
only the two highest-confidence boxes determine hand identity and contact;
all detections are still drawn. References reset for each video run.

`IMAGE_LEFT_HAND=right` is the mirrored-view default in the launcher (or
`--image-left-hand right`). The optional `left` value reverses the pair assignment
for a different camera setup. Check the zone labels in your output. This is
position matching for one operator; crossing hands, large unseen movements, or
multiple operators can still produce incorrect identities.

Each hand is outlined in cyan with horizontal and vertical midpoint lines. The
active quarter is outlined in magenta, turning red when it overlaps a manual.
Only positive-area intersection of that quarter and a manual qualifies; touching
an edge or overlap in another quarter does not count.

## Select and save the counting ROI

The inference batch launcher automatically opens the selector on first use. To
change the saved region later, run
[`select_roi.bat`](select_roi.bat) after editing its `SOURCE`, or from the
repository root:

```bat
projects\ks_honeywell\select_roi.bat "video\input.mp4"
```

Drag one rectangle around the counting workspace and press **Enter** or **Space**
to save. Press **C** to cancel without changing an existing saved selection. The
preview fits within 1280x720, and selected coordinates are mapped back to the
original video resolution. The ROI is saved in `projects/ks_honeywell/roi.json`
with source width/height and `[x1,y1,x2,y2]` coordinates. It can be committed to Git.
Inference draws a yellow `COUNTING ROI` border and rejects a saved ROI whose frame
dimensions differ from the current video. For a different camera framing at the
same resolution, manually select the region again.

For a later preview frame or custom output file, activate `5090` and run:

```bat
python projects\ks_honeywell\select_roi.py --source "video\input.mp4" --frame 100 --output "projects\ks_honeywell\roi.json"
```

Both the **full hand box** and the **manual box** must be fully contained in the
ROI for quadrant/hand-box contact to count. Partial overlap with the ROI does not
qualify, even if the selected hand quarter lies inside. Remembered manual boxes
are checked against the same ROI. All boxes outside remain visible. Hand identity
references use the full frame, so hands outside the ROI can help initialize the
pair but cannot trigger a count. In optional fingertip mode, the manual box must
be inside the ROI. Omitting `--roi` in a direct Python invocation counts over the
full frame; the batch launcher always supplies its saved ROI.

Options: `--imgsz 640`, `--manual-conf 0.25`, `--hand-conf 0.25`,
`--contact-duration 0.2`, `--release-duration 0.3`. `--manual-conf` filters all
custom-model classes. Use `--device cpu` for CPU, `--device mps` for Apple Silicon,
or omit it for Ultralytics' automatic selection. Set contact duration to `0` for
an immediate event on qualifying overlap. Run with `--help` for all arguments.

## Remembered boxes and timing

- The latest detections for each of `manual1`–`manual4` are retained for the entire
  video and refreshed whenever that class is detected again. Missing manual
  detections use these cached boxes for contact checks and drawing, labeled
  `REMEMBERED`. Their confidence is the last detection's score. Other objects and
  hands are not cached. A manual must be detected once before it can be remembered.
- This assumes a fixed camera and stationary manual locations. A hidden manual
  that moves or is removed leaves a stale location until detected again. Restart
  inference for a new scene; there is no automatic cache expiry.
- A touch fires after 0.2 seconds of continuous contact. Continued overlap counts
  once. Rearming requires 0.3 seconds without contact. Missing hands count as no
  contact; missing manuals can still match remembered locations.
- Counts are per manual class, not object instance or operator. Multiple boxes
  of a class share contact state; different manual classes count independently.
- Timing uses decoded frame index / source FPS, starting at zero. Thresholds are
  evaluated at frame timestamps, independently of inference speed.
- Output preserves decoded dimensions and reported FPS using `mp4v`. It writes
  variable-frame-rate input at constant reported FPS; normalize such input first
  if exact timestamps matter. Invalid FPS and existing output files are rejected.

The quadrant is a contact approximation, not an index-fingertip detector. A hand
hovering over a manual can also overlap its box. These public weights have not
been validated on this production line.

## Other contact modes

`--contact-mode hand-box` restores whole-hand overlap and displays PICKUP events.

The previous fingertip mode remains available with `--contact-mode index-finger`.
It requires optional `python -m pip install "mediapipe>=0.10.20,<0.11"` and
[Google's Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
model at `projects/ks_honeywell/weights/hand_landmarker.task` (override using
`--landmark-model`). Download it from Anaconda Prompt:

```bat
curl.exe -fL --retry 3 -o projects\ks_honeywell\weights\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

In that mode, `--num-hands 2` is the landmark hand limit, landmark 8 must lie
inside or on a manual box, and missing landmarks never fall back to box overlap.
These landmark options do not affect the default YOLO quadrant mode.

## Validation

Run the lightweight behavioral tests without installing model dependencies:

```sh
python -m unittest discover -s projects/ks_honeywell -p "test_*.py" -v
```

For real-model validation, use representative footage and your weights. Confirm
all class boxes, side assignment, selected quarters, remembered manual locations,
event timing, counts, duration, dimensions, and FPS in the output MP4. The manual
weights and footage are not bundled. If processing fails, a partial MP4 may remain;
choose a new path or remove the partial output before retrying.
