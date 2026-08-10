from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # Load the exported model.
    # model = YOLO(r"D:\Dre\NK_PDE\logs\0919_dino\0919\exported_models\exported_last.pt")
    model = YOLO(r"yolo26s.pt")

    # Fine-tune with ultralytics.
    # model.train(data=r"D:\Dre\NK_PDE\yolo_dataset\0108_180\data.yaml", epochs=300, project="logs/0109_pt", name="pt_yolo11s_0109_2", workers=0, device='cuda', patience=20)
    model.train(
        data=r"D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\data.yaml",
        project="logs/hph_hw",
        name="yolo26s",
        
        # --- Core Training Params ---
        epochs=500,
        patience=50,      # Increased patience: small objects take longer to converge
        batch=8,          # Lower batch size if you run out of VRAM due to high imgsz
        imgsz=640,       # CRITICAL for screen QC: high res resolves the thin plastic
                          # sheen/wrinkles that distinguish OK vs no-plastic. Infer at 1280 too.
        device='cuda',
        workers=4,        # Set to 4-8 to speed up data loading if CPU allows

        # --- Optimizer & Loss ---
        optimizer='auto', # or 'AdamW' generally works well for small detail
        cos_lr=True,      # Cosine learning rate usually yields better final precision

        # --- Orientation augmentation: OFF for screen QC ---
        # Screens sit in the tray in ONE canonical orientation. A flipped screen (back
        # showing) is a DEFECT, so we must NOT teach the model orientation-invariance.
        # These were copied from a wire task; they were the direct cause of the flipped
        # screen scoring high. Keeping them at 0 makes a flipped screen read as
        # out-of-distribution (confidence drops below the reject threshold).
        degrees=0.0,      # was 45 (wire task) -> screens have a fixed orientation
        flipud=0.0,       # was 0.5 -> vertical flip IS the defect; never augment it
        fliplr=0.0,       # was 0.5 -> tighten the learned front-facing appearance

        # --- Keep the "OK screen" distribution tight (helps no-plastic read as OOD) ---
        mosaic=1.0,       # Keep: general robustness; disabled at the end via close_mosaic
        mixup=0.0,        # was 0.1 -> blending blurs the fine plastic surface cue
        copy_paste=0.0,   # was 0.3 -> pasting screens onto new backgrounds loosens OK dist.

        # --- Fine-Tuning Strategy ---
        close_mosaic=20,  # IMPORTANT: Turn off Mosaic for the last 20 epochs.
                        # This lets the model train on "real", un-shrunk images
                        # to maximize precision on the fine plastic texture.

        # --- Color Augmentation: keep light so the plastic sheen isn't washed out ---
        hsv_h=0.015,      # Slight hue shift
        hsv_s=0.2,        # was 0.7 -> heavy saturation shift washes out the plastic sheen
        hsv_v=0.2,        # was 0.4 -> keep minimal lighting robustness, don't flatten sheen
        scale=0.1,
        erasing=0.0,      # was 0.5 -> random erasing can delete the plastic/no-plastic cue
    )