from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # Load the exported model.
    # model = YOLO(r"D:\Dre\NK_PDE\logs\0919_dino\0919\exported_models\exported_last.pt")
    model = YOLO(r"yolo26n.pt")

    # Fine-tune with ultralytics.
    # model.train(data=r"D:\Dre\NK_PDE\yolo_dataset\0108_180\data.yaml", epochs=300, project="logs/0109_pt", name="pt_yolo11s_0109_2", workers=0, device='cuda', patience=20)
    model.train(
        data=r"D:\Dre\NK_PDE\yolo_dataset\wahoo_0203\data.yaml",
        epochs=300,
        project="logs/0203_pt",
        name="pt_yolo26n_0203",
        workers=0,
        device='cuda',
        patience=30,
        # --- Augmented Parameters ---
        degrees=180,  # Rotates image +/- 180 degrees (covers random 90/180 angles)
        flipud=0.5,   # 50% chance of vertical flip (upside down/180 flip)
        fliplr=0.5    # 50% chance of horizontal flip (standard 90-degree flip logic)
    )