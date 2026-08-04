import cv2
import os
import glob
import json

MASK_FILE = 'hph_hw_masks.json'
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
GLOBAL_OUTPUT_DIR = r'D:\Dre\PDE_yolo_infra\image\masked_hph_hw3'
ROOT_IMAGE_DIR = r'D:\Dre\PDE_yolo_infra\image\hph_hw3'

def save_masks(masks, filename=MASK_FILE):
    with open(filename, 'w') as f:
        json.dump(masks, f, indent=4)
    print(f"✅ Masks saved to {filename}")

def load_masks(filename=MASK_FILE):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def select_masks(image_path):
    """
    Interactive tool to select mask areas.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []

    original_img = img.copy()
    ref_points = []
    saved_masks = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(ref_points) >= 2:
                ref_points = []
            ref_points.append((x, y))
            
            canvas = original_img.copy()
            for m in saved_masks:
                cv2.rectangle(canvas, (m['x1'], m['y1']), (m['x2'], m['y2']), (0, 255, 0), 2)
            for pt in ref_points:
                cv2.circle(canvas, pt, 5, (0, 0, 255), -1)
            if len(ref_points) == 2:
                cv2.rectangle(canvas, ref_points[0], ref_points[1], (0, 0, 255), 2)
            cv2.imshow("Select Masks", canvas)

    cv2.namedWindow("Select Masks", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Masks", mouse_callback)
    cv2.imshow("Select Masks", img)

    print("\n--- Mask Selection Mode ---")
    print("1. Click TWO points to define a rectangle.")
    print("2. Press 's' to SAVE the current rectangle.")
    print("3. Press 'r' to RESET current selection.")
    print("4. Press 'q' to FINISH and save all rectangles.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(ref_points) == 2:
                x1, y1 = ref_points[0]
                x2, y2 = ref_points[1]
                saved_masks.append({
                    'x1': min(x1, x2), 'y1': min(y1, y2),
                    'x2': max(x1, x2), 'y2': max(y1, y2)
                })
                print(f"Added mask: {saved_masks[-1]}")
                ref_points = []
            else:
                print("Select 2 points first!")
        elif key == ord('r'):
            ref_points = []
            print("Selection reset.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return saved_masks

def process_folder(folder_path, masks, output_dir):
    """
    Processes all images in a folder and saves them into the global output_dir.
    Prepends folder name to avoid collisions.
    """
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        return

    # Use the immediate parent folder name to prefix the files
    parent_folder_name = os.path.basename(folder_path)
    
    # Skip if we are inside the output directory itself
    if folder_path.rstrip('\\/') == output_dir.rstrip('\\/'):
        return

    print(f"📁 Processing folder: {folder_path} -> {len(image_paths)} images")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue

        for m in masks:
            cv2.rectangle(img, (m['x1'], m['y1']), (m['x2'], m['y2']), (0, 0, 0), -1)

        filename = os.path.basename(img_path)
        # Create a unique filename: folderName_originalName.jpg
        unique_filename = f"{parent_folder_name}_{filename}"
        cv2.imwrite(os.path.join(output_dir, unique_filename), img)

def main():
    root_image_dir = ROOT_IMAGE_DIR
    
    # 1. Ensure Global Output Folder exists
    if not os.path.exists(GLOBAL_OUTPUT_DIR):
        os.makedirs(GLOBAL_OUTPUT_DIR)
        print(f"Created output directory: {GLOBAL_OUTPUT_DIR}")

    # 2. Load or Select Masks
    masks = load_masks()
    
    # Find a sample image to show the selector if needed
    sample_img = None
    for root, dirs, files in os.walk(root_image_dir):
        # Skip the masked folder during search
        if os.path.abspath(root) == os.path.abspath(GLOBAL_OUTPUT_DIR):
            continue
        for f in files:
            if any(f.lower().endswith(ext.strip('*.')) for ext in IMAGE_EXTENSIONS):
                sample_img = os.path.join(root, f)
                break
        if sample_img: break

    if not masks:
        print("No masks found.")
        if sample_img:
            masks = select_masks(sample_img)
            save_masks(masks)
        else:
            print("No images found to select masks from.")
            return
    else:
        print(f"Loaded {len(masks)} masks from {MASK_FILE}")
        choice = input("Update masks? (y/n): ").lower()
        if choice == 'y':
            if sample_img:
                masks = select_masks(sample_img)
                save_masks(masks)
            else:
                print("No images found to update masks.")

    if not masks:
        print("No masks defined. Exiting.")
        return

    # 3. Walk through all subfolders
    print(f"\n🚀 Starting batch processing. All images will be saved to: {GLOBAL_OUTPUT_DIR}")
    for root, dirs, files in os.walk(root_image_dir):
        # Skip the output directory itself to avoid infinite loops/re-masking
        if os.path.abspath(root) == os.path.abspath(GLOBAL_OUTPUT_DIR):
            continue
        
        process_folder(root, masks, GLOBAL_OUTPUT_DIR)
    
    print("\n✅ All images processed and saved to the 'masked' folder.")

if __name__ == "__main__":
    main()
