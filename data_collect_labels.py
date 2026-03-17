import os
import shutil
import glob

def move_labeled_data(source_base_dir, target_base_dir):
    """
    Finds all folders in source_base_dir and moves their images and labels
    to the target train structure.
    """
    target_images_dir = os.path.join(target_base_dir, 'images', 'train')
    target_labels_dir = os.path.join(target_base_dir, 'labels', 'train')

    # Create target directories if they don't exist
    print("Ensuring target directories exist:")
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    total_images = 0
    total_labels = 0

    # Get all subdirectories in the source directory
    subdirectories = [os.path.join(source_base_dir, d) for d in os.listdir(source_base_dir) 
                      if os.path.isdir(os.path.join(source_base_dir, d))]

    for folder in subdirectories:
        # Skip the target folder itself
        if os.path.abspath(folder) == os.path.abspath(target_base_dir):
            continue
            
        print(f"\nScanning folder: {folder}")
        
        # Recursive search for all files
        files = glob.glob(os.path.join(folder, '**', '*.*'), recursive=True)
        
        img_count = 0
        lbl_count = 0

        for file_path in files:
            if os.path.isdir(file_path):
                continue

            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            
            # Handle images
            if ext in image_extensions:
                dest_path = os.path.join(target_images_dir, filename)
                try:
                    shutil.move(file_path, dest_path)
                    img_count += 1
                except Exception as e:
                    print(f"Error moving image {filename}: {e}")
            
            # Handle YOLO labels (.txt)
            elif ext == '.txt':
                # Skip non-label files
                if filename.lower() in ['classes.txt', 'notes.txt', 'data.yaml', 'train.txt', 'val.txt']:
                    continue
                    
                dest_path = os.path.join(target_labels_dir, filename)
                try:
                    shutil.move(file_path, dest_path)
                    lbl_count += 1
                except Exception as e:
                    print(f"Error moving label {filename}: {e}")

        print(f"  Moved {img_count} images and {lbl_count} labels from {os.path.basename(folder)}")
        total_images += img_count
        total_labels += lbl_count

    print("\nOperation complete.")
    print(f"Total images moved: {total_images}")
    print(f"Total labels moved: {total_labels}")

if __name__ == "__main__":
    # The parent directory containing all labeled data folders
    source_root = r'D:\Dre\PDE_yolo_infra\train_dataset\source'
    
    # Target dataset directory
    target_root = r'D:\Dre\PDE_yolo_infra\train_dataset\wahoo_train_0311'
    
    move_labeled_data(source_root, target_root)
