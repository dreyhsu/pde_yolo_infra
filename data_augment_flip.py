import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

def flip_yolo_annotations(input_path, output_path):
    """
    Flip YOLO annotations 180 degrees.
    x_new = 1.0 - x_center
    y_new = 1.0 - y_center
    """
    if not os.path.exists(input_path):
        return False
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    flipped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Flip 180 degrees
        x_new = 1.0 - x
        y_new = 1.0 - y
        
        flipped_lines.append(f"{cls} {x_new:.6f} {y_new:.6f} {w:.6f} {h:.6f}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(flipped_lines) + '\n')
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Flip YOLO dataset 180 degrees (Horizontal + Vertical)")
    parser.add_argument("--input_dir", type=str, default=r"D:\Dre\PDE_yolo_infra\train_dataset\wahoo_0316_aug", help="Path to the dataset root (containing images and labels folders)")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the flipped dataset (default: input_dir + '_flipped_180')")
    
    args = parser.parse_args()
    
    input_root = Path(args.input_dir)
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = input_root.parent / (input_root.name + "_flipped_180")
    
    img_in_root = input_root / "images"
    lbl_in_root = input_root / "labels"
    
    img_out_root = output_root / "images"
    lbl_out_root = output_root / "labels"
    
    # Supported image extensions
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all image files recursively
    image_files = []
    for ext in img_exts:
        image_files.extend(list(img_in_root.rglob(f"*{ext}")))
    
    print(f"Processing {len(image_files)} images from {img_in_root}...")
    
    for img_path in tqdm(image_files):
        # 1. Flip Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Flip both axes (180 degrees)
        flipped_img = cv2.flip(img, -1)
        
        # Calculate relative path to maintain structure
        rel_path = img_path.relative_to(img_in_root)
        out_img_path = img_out_root / rel_path
        
        # Ensure output subdirectory exists
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save flipped image
        cv2.imwrite(str(out_img_path), flipped_img)
        
        # 2. Flip Label
        # Find corresponding label path
        lbl_rel_path = rel_path.with_suffix(".txt")
        lbl_path = lbl_in_root / lbl_rel_path
        out_lbl_path = lbl_out_root / lbl_rel_path
        
        # Ensure output label directory exists
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        
        if lbl_path.exists():
            flip_yolo_annotations(str(lbl_path), str(out_lbl_path))
        else:
            # Create empty label if original doesn't exist
            open(out_lbl_path, 'a').close()

    print(f"Done! Flipped dataset saved to: {output_root}")

if __name__ == "__main__":
    main()
