import os
import sys

TARGET_PATH = r"train_dataset\wahoo_0209\acc_0211"

def rename_files(target_path):
    """
    Renames .jpg and .txt files in the target directory by prepending 
    the directory name to the filename.
    """
    # Normalize path for the current OS
    target_path = os.path.normpath(target_path)
    
    if not os.path.isdir(target_path):
        print(f"Error: {target_path} is not a valid directory.")
        return

    # Use the last part of the path as the prefix
    prefix = os.path.basename(target_path)
    print(f"Using prefix: {prefix}_")

    count = 0
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(target_path):
        for filename in files:
            # Check for .jpg and .txt files
            if filename.lower().endswith(('.jpg', '.txt')):
                # Skip if the prefix is already present to avoid double renaming
                if filename.startswith(f"{prefix}_"):
                    continue
                
                old_file_path = os.path.join(root, filename)
                new_filename = f"{prefix}_{filename}"
                new_file_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_file_path, new_file_path)
                    count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")

    print(f"Successfully renamed {count} files.")

if __name__ == "__main__":
    rename_files(TARGET_PATH)

