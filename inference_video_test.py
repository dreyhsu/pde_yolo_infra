from ultralytics import YOLO
import os

def run_inference():
    # Model path as specified by the user
    model_path = r"D:\Dre\PDE_yolo_infra\model\hph_hw.pt"
    # model_path = r"D:\Dre\PDE_yolo_infra\logs\hph_hw\pt_yolo_precision_run\weights\best.pt"
    # Video path as specified by the user
    video_path = r"D:\Dre\PDE_yolo_infra\video\hph_honeywell\ng_test\2026-06-02_070640\screen_flip.mp4"

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
        conf=0.3,        # Confidence threshold
        save=True,        # Save the result
        device='cuda',    # Use GPU if available, otherwise change to 'cpu'
        show=False        # Set to True if you want to see the live window
    )

    print("Inference complete. Results saved in 'runs/detect/predict'.")

if __name__ == "__main__":
    # Fix for multi-processing issues on Windows if needed
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_inference()
