"""
YOLO26-s training for the hph_hw packaging-process detector (9 classes).

Differs from train_yolo.py (yolo11n) in more than the weights file: the whole
augmentation strategy is inverted, because the job of the detector changed.

    OLD (train_yolo.py): YOLO was expected to separate OK from NG by itself, so
      every augmentation that could make a flipped or bare screen look "in
      distribution" was zeroed. Confidence WAS the verdict.

    NOW: screen_qc.py makes the OK/NG call from pixels, using the box geometry
      only as a crop. The detector must therefore fire on ALL THREE scenarios
      (normal / plastic_ng / flip_ng) as reliably as possible. A miss is now the
      worst outcome -- no box means no QC call means a silent pass.

Two consequences drive every value below:

  1. RECALL over selectivity. Augmentation is loosened back up. Orientation and
     colour jitter no longer risk "teaching away" a defect cue, because no defect
     cue lives in the detector any more.

  2. BOX TIGHTNESS is the real metric. screen_qc.py derives its core crop
     (middle 40%) and its illumination ring (the 6%-20% annulus) purely from the
     predicted box. A loose or drifting box slides the core off the screen face
     and pulls non-tray pixels into the ring, which corrupts dark_ratio and
     mark_frac directly. Judge this model on mAP50-95, never mAP50 -- an IoU-0.5
     box would wreck the crop geometry while scoring "fine".

Note that imgsz no longer needs to resolve the plastic sheen: screen_qc.py reads
r.orig_img, the full-resolution frame, not the letterboxed training tensor. 640 is
here for localisation accuracy alone.

Hyperparameters follow the published YOLO26 recipe for the S size
(https://docs.ultralytics.com/guides/yolo26-training-recipe/), adjusted for a
~1500-image fine-tune rather than a from-scratch COCO run.

Requires ultralytics >= 8.4 (the 8.3.x line cannot parse the yolo26 graph).
"""

from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    model = YOLO(r"yolo26s.pt")

    model.train(
        data=r"D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\data.yaml",
        project="logs/hph_hw",
        name="yolo26s",

        # --- Core training params ---
        epochs=150,       # was 500. 1500 images fine-tuning from COCO weights does not
                          # need a from-scratch budget; the recipe's own fine-tune note
                          # is ~50 epochs for <1k images. 150 + early stop is generous.
        patience=30,
        batch=16,         # yolo26s @640 fits 8GB on the 3070. Drop to 8 on OOM.
                          # Bigger is better here: ultralytics accumulates to nbs=64,
                          # so batch=8 means an 8-step accumulation and noisier updates.
        imgsz=640,
        device='cuda',
        workers=4,

        # --- Optimizer -------------------------------------------------------
        # YOLO26's released checkpoints were trained with MuSGD (SGD+Muon), and the
        # LRs below are tuned for it. 'auto' is left here because it always runs --
        # but CHECK the `optimizer:` line the trainer prints on startup. If it did
        # not select MuSGD, set it explicitly. To see what your build accepts:
        #   python -c "import inspect;from ultralytics.engine.trainer import BaseTrainer;print(inspect.getsource(BaseTrainer.build_optimizer))"
        optimizer='auto',
        cos_lr=True,

        lr0=0.001,        # recipe yolo26s is 0.00038 @ batch 128 for a 500ep scratch run.
                          # 0.001 is the recipe's own small-dataset fine-tune value.
        lrf=0.05,         # recipe uses 0.882 (gentle, long schedule). We decay hard
                          # instead: a low terminal LR is what sharpens box regression,
                          # which is the only thing this model is judged on.
        momentum=0.948,   # recipe S
        weight_decay=0.00027,  # recipe S
        warmup_epochs=3.0,

        # --- Loss weights: recipe values for yolo26s, NOT the yolo11 defaults ---
        box=9.83,         # default 7.5. The single most important knob here --
                          # see the box-tightness note in the docstring.
        cls=0.65,         # default 0.5. 9 classes, so this still matters.
        dfl=0.96,         # default 1.5. YOLO26 drops Distribution Focal Loss; this
                          # is the residual regression term, not the old distributional one.

        # --- Orientation: re-enabled, with the old rationale retired ---
        degrees=3.0,      # was 0.0. Small tray skew is real on the line, and box
                          # robustness to it is now worth more than orientation
                          # sensitivity. Kept small: rotating an axis-aligned box
                          # inflates it, which costs tightness.
        fliplr=0.5,       # was 0.0 ("vertical flip IS the defect"). That reasoning
                          # belonged to the old strategy. Mirroring is free
                          # regularisation on a 1500-image set now.
                          # !! Set back to 0.0 if ANY of your other 8 classes is
                          #    defined by left/right orientation (e.g. a directional
                          #    label or an asymmetric connector).
        flipud=0.0,       # Still off, but for a new reason: the camera is fixed and
                          # never sees an inverted tray. Not harmful, just wasted.

        # --- Scale / translate: middle ground ---
        scale=0.3,        # was 0.1 (lock the OK distribution), recipe S uses 0.9 (COCO).
                          # Fixed camera means near-constant object scale, but some
                          # jitter measurably improves box regression. 0.3 splits it.
        translate=0.1,

        # --- Mosaic ---
        mosaic=1.0,       # recipe S is 0.992. Good regularisation for a small set.
        close_mosaic=20,  # recipe says 10; 20 kept from the yolo11 config. Mosaic
                          # actively degrades box precision, so a longer clean tail
                          # is worth it when tightness is the deliverable.
        mixup=0.0,
        copy_paste=0.0,

        # --- Colour: loosened, because the sheen is no longer YOLO's problem ---
        hsv_h=0.015,
        hsv_s=0.4,        # was 0.2 ("washes out the plastic sheen"). screen_qc.py reads
        hsv_v=0.4,        # was 0.2. the sheen from orig_img now, so lighting robustness
                          # in the detector is a pure win.

        plots=True,
    )
