import os
import random
import shutil
import sys

TARGET_PATH = "train_dataset\wahoo_0209"

def split_dataset(target_path, val_ratio=0.2):
    """
    Randomly selects a percentage of images and labels from train folders 
    and moves them to val folders.
    """
    # Define paths
    images_train_dir = os.path.join(target_path, 'images', 'train')
    labels_train_dir = os.path.join(target_path, 'labels', 'train')
    images_val_dir = os.path.join(target_path, 'images', 'val')
    labels_val_dir = os.path.join(target_path, 'labels', 'val')

    # Validate train directories
    if not os.path.exists(images_train_dir) or not os.path.exists(labels_train_dir):
        print(f"Error: Could not find train directories in {target_path}")
        return

    # Create val directories if they don't exist
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(images_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in images/train.")
        return

    # Determine number of files to move
    num_val = int(len(image_files) * val_ratio)
    val_images = random.sample(image_files, num_val)

    print(f"Moving {num_val} files (approx. {val_ratio*100}%) to val...")

    count = 0
    for img_name in val_images:
        # Define file names
        label_name = os.path.splitext(img_name)[0] + '.txt'
        
        src_img = os.path.join(images_train_dir, img_name)
        dst_img = os.path.join(images_val_dir, img_name)
        
        src_label = os.path.join(labels_train_dir, label_name)
        dst_label = os.path.join(labels_val_dir, label_name)

        # Move image
        try:
            shutil.move(src_img, dst_img)
            
            # Move label if it exists
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f"Warning: Label not found for {img_name}")
            
            count += 1
        except Exception as e:
            print(f"Failed to move {img_name}: {e}")

    print(f"Successfully moved {count} images and their labels to val.")

if __name__ == "__main__":
    split_dataset(TARGET_PATH)
