import cv2
import os
import argparse
import glob

def draw_yolo_annotations(image_path, label_path):
    """
    Reads a YOLO image and label, then returns the image with bounding boxes.
    """
    if not os.path.exists(image_path):
        return None
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w, _ = img.shape

    # Load labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{int(class_id)}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(img, "NO LABEL FOUND", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img

def get_label_path(image_path):
    """
    Infers the label path from the image path by replacing 'images' with 'labels'
    and changing the extension to .txt.
    """
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    
    # Standard YOLO structure: labels is sibling to images
    label_dir = image_dir.replace('images', 'labels')
    label_name = os.path.splitext(image_name)[0] + '.txt'
    
    return os.path.join(label_dir, label_name)

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations.")
    parser.add_argument("input", help="Path to an image file or a directory of images")
    args = parser.parse_args()

    # Get list of images
    if os.path.isdir(args.input):
        image_list = sorted(glob.glob(os.path.join(args.input, "*.jpg")) + 
                            glob.glob(os.path.join(args.input, "*.png")))
    else:
        image_list = [args.input]

    if not image_list:
        print("No images found.")
        return

    idx = 0
    while True:
        image_path = image_list[idx]
        label_path = get_label_path(image_path)
        
        print(f"[{idx+1}/{len(image_list)}] Displaying: {os.path.basename(image_path)}")
        
        display_img = draw_yolo_annotations(image_path, label_path)
        
        if display_img is not None:
            # Resize for display if too large (optional)
            h, w = display_img.shape[:2]
            if h > 1000 or w > 1600:
                scale = min(1000/h, 1600/w)
                display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)))

            cv2.imshow("YOLO Visualization (n: next, p: prev, q: quit)", display_img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(image_list)
        elif key == ord('p'):
            idx = (idx - 1) % len(image_list)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
