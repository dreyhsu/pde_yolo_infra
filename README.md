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
5. **Split Dataset:** Create your validation set with `data_split_val.py`.
6. **Train:** Start training your model with `train_yolo.py`.

---

## 📂 Script Reference

### 🛠️ Preparation (`prep_`)
- `prep_cut_video.py`: Batch processes videos based on timestamps in a CSV.
- `prep_extract_frames.py`: Interactive tool for ROI extraction and frame generation.
- `prep_apply_masks.py`: Interactively black out specific regions across a frame set.
- `prep_select_roi.py`: Manually select and save ROI coordinates from images.
- `prep_convert_video.py`: Re-encodes videos to H.264 or resizes them.

### 🧠 Inference (`inference_`)
- `inference_cvat_prelabel.py`: Runs YOLO and exports a ZIP for CVAT's YOLO 1.1 format.
- `inference_video_test.py`: Runs inference on a video and saves the annotated results.

### 📊 Data Management (`data_`)
- `data_collect_labels.py`: Organizes labels and images into standard YOLO directory structure.
- `data_rename_unique.py`: Prepends folder names to files to ensure unique filenames.
- `data_augment_yolo.py`: CLI for YOLO/VOC dataset augmentation with bounding box syncing.
- `data_augment_images.py`: Quick batch image-only augmentation using Albumentations.
- `data_split_val.py`: Randomly splits data into training and validation sets.

### 🏎️ Training (`train_`)
- `train_yolo.py`: Trains or fine-tunes a YOLO model using Ultralytics.

---

## ⚙️ Requirements
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- Albumentations
- FFmpeg (for video processing)
