from ultralytics import YOLO
import os

def run_inference():
    # Model path as specified by the user
    model_path = r"D:\Dre\PDE_yolo_infra\logs\0316_11s_640\pt_yolo_precision_run2\weights\best.pt"
    # model_path = r"D:\Dre\PDE_yolo_infra\logs\0223_11s_640\pt_yolo_precision_run2\weights\best.pt"
    # Video path as specified by the user
    video_path = r"clip/sticker_5.mp4"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Load the YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Run inference on the video
    # 'save=True' will save the output video to 'runs/detect/predict' by default
    print(f"Running inference on: {video_path}")
    results = model.predict(
        source=video_path,
        conf=0.20,        # Confidence threshold
        save=True,        # Save the result
        device='cuda',    # Use GPU if available, otherwise change to 'cpu'
        show=False        # Set to True if you want to see the live window
    )

    print("Inference complete. Results saved in 'runs/detect/predict'.")

if __name__ == "__main__":
    # Fix for multi-processing issues on Windows if needed
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_inference()
