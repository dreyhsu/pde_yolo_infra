import os
import zipfile
from ultralytics import YOLO
from pathlib import Path

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def run_inference_for_cvat():
    # 1. Configuration
    model_path = r"D:\Dre\PDE_yolo_infra\logs\0313_11s_640\pt_yolo_precision_run\weights\yolo11s_640_0311.pt"
    image_dir = r"D:\Dre\PDE_yolo_infra\image\masked"
    output_dir = "cvat_yolo_export"
    obj_train_data_dir = os.path.join(output_dir, "obj_train_data")
    iou_threshold = 0.5  # Threshold to define "overlapping"
    
    # List of filename prefixes to skip inference
    prefixes_to_skip = [
        "jona_holdings_jona_holdings_frame",
    ]
    
    # 2. Setup output directories
    if not os.path.exists(obj_train_data_dir):
        os.makedirs(obj_train_data_dir)
    
    # 3. Load Model
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = YOLO(model_path)
    
    # Get class names
    class_names = model.names
    num_classes = len(class_names)
    
    # 4. Create obj.names
    obj_names_path = os.path.join(output_dir, "obj.names")
    with open(obj_names_path, "w") as f:
        for i in range(num_classes):
            f.write(f"{class_names[i]}\n")
    
    # 5. Create obj.data
    obj_data_path = os.path.join(output_dir, "obj.data")
    with open(obj_data_path, "w") as f:
        f.write(f"classes = {num_classes}\n")
        f.write("train = train.txt\n")
        f.write("names = obj.names\n")
        f.write("backup = backup/\n")
    
    # 6. Run Inference and Save Annotations
    image_list = []
    print(f"Running inference on images in: {image_dir}")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    total_images = len(images)
    print(f"Found {total_images} images.")
    
    for i, img_name in enumerate(images):
        if any(img_name.startswith(prefix) for prefix in prefixes_to_skip):
            continue

        img_path = os.path.join(image_dir, img_name)
        results = model(img_path, conf=0.25, verbose=False)
        
        txt_name = Path(img_name).stem + ".txt"
        txt_path = os.path.join(obj_train_data_dir, txt_name)
        
        # Filter bboxes: pick smallest among overlaps
        kept_boxes = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            box_data = []
            for j in range(len(boxes)):
                xyxy = boxes.xyxy[j].tolist()
                cls = int(boxes.cls[j].item())
                xywhn = boxes.xywhn[j].tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                box_data.append({'xyxy': xyxy, 'cls': cls, 'xywhn': xywhn, 'area': area})
            
            # Sort by area (smallest first)
            box_data.sort(key=lambda x: x['area'])
            
            for b in box_data:
                is_overlapping_with_smaller = False
                for kept in kept_boxes:
                    if calculate_iou(b['xyxy'], kept['xyxy']) > iou_threshold:
                        is_overlapping_with_smaller = True
                        break
                if not is_overlapping_with_smaller:
                    kept_boxes.append(b)
        
        # Write annotations to file
        with open(txt_path, "w") as f:
            for b in kept_boxes:
                # YOLO format: cls x_center y_center width height (normalized)
                line = f"{b['cls']} {' '.join(map(str, b['xywhn']))}\n"
                f.write(line)
            
        image_list.append(f"obj_train_data/{img_name}")
            
        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"Processed {i + 1}/{total_images} images...")

    # 7. Create train.txt
    train_txt_path = os.path.join(output_dir, "train.txt")
    with open(train_txt_path, "w") as f:
        for line in image_list:
            f.write(f"{line}\n")
            
    # 8. Create ZIP file for CVAT
    zip_path = "cvat_annotations.zip"
    print(f"Creating ZIP file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add obj.names
        zipf.write(obj_names_path, "obj.names")
        # Add obj.data
        zipf.write(obj_data_path, "obj.data")
        # Add train.txt
        zipf.write(train_txt_path, "train.txt")
        # Add all txt files in obj_train_data/
        for root, _, files in os.walk(obj_train_data_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join("obj_train_data", file))
                
    print(f"Done! You can now upload '{zip_path}' to CVAT using the 'YOLO 1.1' annotation format.")

if __name__ == "__main__":
    run_inference_for_cvat()
