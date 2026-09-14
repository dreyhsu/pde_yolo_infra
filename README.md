# PDE YOLO Infrastructure

A specialized toolkit for preparing, labeling, augmenting, and training YOLO-based object detection models, specifically optimized for PDE workflows.

---

## 🚀 Workflows

### 1. Data Preparation for Labeling
Convert raw video segments into cropped, masked frames ready for annotation.
1. **Cut Clips:** Use `prep_cut_video.py` with `crop_task.csv` to extract short segments.
2. **Extract Frames:** Run `prep_extract_frames.py` to extract and crop frames with specific ROIs.
3. **Select ROIs:** Use `prep_select_roi.py` for manual region selection if needed.
4. **Apply Masks:** (Optional) Mask sensitive info or static noise with `prep_apply_masks.py`.
5. **Convert Format:** Ensure video compatibility with `prep_convert_video.py`.
6. **Pre-process Source (Optional):** Flip upside-down recordings with `prep_flip_video.py` or join split recordings with `prep_concat_videos.py` before extraction.

### 2. AI-Assisted Labeling (Inference)
Speed up the annotation process by using existing models to pre-label data.
1. **Pre-labeling:** Run `inference_cvat_prelabel.py` to generate YOLO 1.1 annotations.
2. **CVAT Integration:** Zip the output and upload it to CVAT via "Upload Annotations".
3. **Verification:** Use `inference_video_test.py` to test your model's performance on a full video.

### 3. Dataset Management & Training
Organize, expand, and train on your labeled data.
1. **Collect Labels:** Consolidate labels from CVAT into your dataset folder with `data_collect_labels.py`.
2. **Rename Files:** Prevent collisions when merging datasets with `data_rename_unique.py`.
3. **Augment YOLO:** Boost your dataset size/diversity with `data_augment_yolo.py`.
4. **Augment Images:** Perform pure image augmentation with `data_augment_images.py`.
5. **Flip 180:** Augment dataset by flipping 180 degrees with `data_augment_flip.py`.
6. **Visualize Labels:** Verify your annotations with `data_visualize.py`.
7. **Split Dataset:** Create your validation set with `data_split_val.py`.
8. **Train:** Start training your model with `train_yolo.py`.

### 4. Model Evaluation
Compare many models against many videos using human-labeled temporal ground truth.
Everything lives under `projects/<name>/`, so one checkout can serve several projects.

1. **Init:** `python eval_init_project.py --name hph_hw --classes-from <classes.txt | data.yaml | best.pt>`
   then fill in `models[]` and `videos[]` in `projects/hph_hw/config.json` (use forward slashes).
2. **Cache detections:** `python eval_run_detect.py --project projects/hph_hw`
   Runs each model over each video **once** at `cache_conf` (0.01) and stores every raw box.
   Do this first — it validates all paths and reports class-set mismatches before you spend
   time labeling. Re-runs are skipped unless the weights, video, or settings changed.
3. **Label ground truth:** `python eval_label_timeline.py --project projects/hph_hw --video test1`
   Draw ROIs, scrub the video, mark the frame range where each class is visible in an ROI and
   type its name and instance count. Several classes may be active at once. Press `h` for keys.
   All input happens **inside the video window** — the terminal is only used for startup logs.
4. **Report:** `python eval_report.py --project projects/hph_hw`
   Sweeps confidence thresholds offline (no re-inference), writes metric CSVs, a
   `leaderboard.md`, and the Gantt timeline chart.
5. **Re-chart only:** `python generate_eval_gantt.py --project projects/hph_hw --video test1 --roi tray --conf 0.4`

**What the metrics mean**
- *Frame level* — per-class precision/recall/F1 where each frame is a multiset of instances, so
  predicting one screen where you labeled two is penalized. Plus a confusion matrix.
- *Segment level* — temporal-IoU-matched F1 at 0.1/0.25/0.5, matched per class, plus an edit
  score over the start-ordered sequence of class tokens ("did it see the right sequence of steps").
- *Timing* — onset latency to the first stable detection, and flicker (raw fragments per GT
  segment; 1.0 is perfect). Both are the most actionable numbers for tuning `conf`.
- *Counting* — MAE against the instance counts you typed while labeling.

**Reading the chart** — one row per model plus a GT row, one lane per class within each row.
Color identifies the class; a washed-out bar means the model was less confident over that run.

> `projects/` is gitignored **except** `config.json`, `classes.txt` and `gt/*.json`, so your
> hand-labeled ground truth survives a clean checkout while caches and reports stay local.

### 5. OpenCV OK/NG Template Matching
Use `template_match_video.py` to compare OK and NG templates in every frame. Matching is
restricted to the regions in `saved_rois.txt`. A frame is `REAL OK` only when the OK
confidence is above the threshold and higher than the NG confidence; otherwise it is `NG`.

Install OpenCV if it is not already available:

```bash
python3 -m pip install opencv-python
```

Run the script with a template image, input video, and output video:

```bash
conda run -n tf python template_match_video.py \
  transfer/nk_apl_mcio/images/template.png \
  transfer/nk_apl_mcio/images/ng.png \
  path/to/input.mp4 \
  path/to/output.mp4 \
  --threshold 0.75
```

Add `--display` to preview the annotated frames while processing. Press `q` in the
preview window to stop early.

- Green box (`REAL OK`): OK confidence is above the threshold and above NG confidence.
- Red box (`NG`): the `REAL OK` rule was not satisfied.
- Blue box: ROI loaded from `saved_rois.txt`; template matching searches only inside it.
- The label displays both OK and NG confidence scores on every frame.
- Both templates should be tightly cropped and approximately the same size and orientation
  as their corresponding targets in the video.
- The generated video does not include the source audio.

---

## 📂 Script Reference

### 🛠️ Preparation (`prep_`)
- `prep_cut_video.py`: Batch processes videos based on timestamps in a CSV.
- `prep_extract_frames.py`: Interactive tool for ROI extraction and frame generation.
- `prep_apply_masks.py`: Interactively black out specific regions across a frame set.
- `prep_select_roi.py`: Manually select and save ROI coordinates from images.
- `prep_convert_video.py`: Re-encodes videos to H.264 or resizes them.
- `prep_flip_video.py`: Flips a single video 180° (horizontal + vertical) using FFmpeg.
- `prep_concat_videos.py`: Concatenates two videos (fast stream-copy, with a robust re-encode fallback via `--reencode`).

### 🧠 Inference (`inference_`)
- `inference_cvat_prelabel.py`: Runs YOLO and exports a ZIP for CVAT's YOLO 1.1 format.
- `inference_video_test.py`: Runs inference on a video and saves the annotated results.
- `template_match_video.py`: Compares OK and NG templates inside saved ROIs and writes the
  winning bounding box, both confidence scores, and the `REAL OK`/`NG` result to a video.

### 📊 Data Management (`data_`)
- `data_collect_labels.py`: Organizes labels and images into standard YOLO directory structure.
- `data_rename_unique.py`: Prepends folder names to files to ensure unique filenames.
- `data_augment_yolo.py`: CLI for YOLO/VOC dataset augmentation with bounding box syncing.
- `data_augment_images.py`: Quick batch image-only augmentation using Albumentations.
- `data_augment_flip.py`: Flips an entire YOLO dataset (images and labels) 180 degrees. Should modify the python record script, flip the record video. 
- `data_visualize.py`: Interactive tool to browse and verify YOLO annotations on images.
- `data_split_val.py`: Randomly splits data into training and validation sets.

### 🏎️ Training (`train_`)
- `train_yolo.py`: Trains or fine-tunes a YOLO model using Ultralytics.

### 📏 Evaluation (`eval_`)
- `eval_init_project.py`: Scaffolds `projects/<name>/` and seeds `classes.txt` + `config.json`.
- `eval_label_timeline.py`: Interactive OpenCV tool for labeling temporal ground truth inside ROIs.
- `eval_run_detect.py`: Caches every raw detection per model × video so confidence can be swept offline.
- `eval_report.py`: Scores the cache against ground truth; writes metric CSVs and `leaderboard.md`.
- `eval_common.py`: Shared library (config/GT/cache IO, ROI filter, segment math). Not run directly.

### 📈 Analysis / Visualization (`generate_`)
- `generate_eval_gantt.py`: Gantt timeline of predictions vs ground truth, dimmed by confidence. Driven by `eval_report.py`, or run standalone to re-render.
- `generate_plotly_chart.py`: Mock-data layout demo for the original single-label Gantt style. Not wired to real inference — use `generate_eval_gantt.py` for that.

---

## ⚙️ Requirements
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- Albumentations
- Plotly & pandas (for analysis/visualization)
- Kaleido (optional — only for PNG export of eval charts; the HTML always works).
  Plotly 6.x needs `pip install -U kaleido`; Plotly 5.x needs `pip install kaleido==0.2.1`.
- FFmpeg (for video processing)
