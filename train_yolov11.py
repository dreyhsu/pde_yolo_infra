from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # Load the exported model.
    # model = YOLO(r"D:\Dre\NK_PDE\logs\0919_dino\0919\exported_models\exported_last.pt")
    model = YOLO(r"yolo26m.pt")

    # Fine-tune with ultralytics.
    # model.train(data=r"D:\Dre\NK_PDE\yolo_dataset\0108_180\data.yaml", epochs=300, project="logs/0109_pt", name="pt_yolo11s_0109_2", workers=0, device='cuda', patience=20)
    model.train(
        data=r"train_dataset\wahoo_0209\data.yaml",
        project="logs/0211_pt",
        name="pt_yolo_precision_run",
        
        # --- Core Training Params ---
        epochs=300,
        patience=50,      # Increased patience: small objects take longer to converge
        batch=8,          # Lower batch size if you run out of VRAM due to high imgsz
        imgsz=1280,       # CRITICAL: High res to resolve the thin wires/loops clearly
        device='cuda',
        workers=4,        # Set to 4-8 to speed up data loading if CPU allows
        
        # --- Optimizer & Loss ---
        optimizer='auto', # or 'AdamW' generally works well for small detail
        cos_lr=True,      # Cosine learning rate usually yields better final precision
        
        # --- Augmentations for Flexible/Wire Objects ---
        degrees=180,      # Perfect for wires (they can be in any orientation)
        flipud=0.5,       # Keep: Wires have no "up" or "down"
        fliplr=0.5,       # Keep: Wires have no "left" or "right"
        
        mosaic=1.0,       # Keep: Good for general robustness
        mixup=0.1,        # Low mixup: Helps, but too much confuses small boundaries
        copy_paste=0.3,   # OPTIONAL: Great for small objects (pastes targets onto new backgrounds)
        
        # --- Fine-Tuning Strategy ---
        close_mosaic=20,  # IMPORTANT: Turn off Mosaic for the last 20 epochs. 
                        # This lets the model train on "real", un-shrunk images 
                        # to maximize precision on the small connectors.
        
        # --- Color Augmentation (Handle lighting changes) ---
        hsv_h=0.015,      # Slight hue shift
        hsv_s=0.7,        # Saturation variety (wires might look different under lights)
        hsv_v=0.4,        # Value (brightness) variety
    )