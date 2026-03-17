import csv
import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
INPUT_CSV = 'crop_task.csv'
VIDEO_DIR = 'video/wahoo'
CLIP_DIR = 'clip'
DEFAULT_EXTENSION = '.mp4'

def time_to_seconds(time_str):
    """Converts MM:SS or HH:MM:SS string to total seconds (float)."""
    if not time_str:
        return 0.0
    try:
        parts = list(map(float, time_str.split(':')))
        if len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            return float(time_str) # Just seconds
    except ValueError:
        print(f"Warning: Could not parse time '{time_str}', assuming 0.")
        return 0.0

def get_video_path(base_name, search_dir):
    search_path = Path(search_dir)
    for file_path in search_path.glob(f"{base_name}.*"):
        if file_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            return file_path
    return None

def cut_video(input_path, output_path, start_str, end_str):
    """
    Calculates duration and cuts video using FFmpeg input seeking.
    """
    
    # 1. Calculate Duration
    start_sec = time_to_seconds(start_str)
    end_sec = time_to_seconds(end_str)
    
    duration = end_sec - start_sec
    
    if duration <= 0:
        print(f"   ⚠ Skipping: End time ({end_str}) is not after Start time ({start_str})")
        return False

    # 2. Build Command
    # Note: We use -ss BEFORE -i for fast seek, so we must use -t (duration) not -to
    cmd = [
        'ffmpeg',
        '-ss', str(start_sec),      # Seek to start
        '-i', str(input_path),      # Input file
        '-t', str(duration),        # Record for this duration
        '-q:v', '1',                # High quality video
        '-c:a', 'copy',             # Copy audio
        '-y',                       # Overwrite
        str(output_path)
    ]

    # Display names clearly
    in_name = input_path.name if isinstance(input_path, Path) else os.path.basename(input_path)
    out_name = output_path.name if isinstance(output_path, Path) else os.path.basename(output_path)

    print(f"   Processing: {in_name} -> {out_name}")
    print(f"   Cut: {start_str} to {end_str} (Duration: {duration:.2f}s)")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ Success")
            return True
        else:
            print(f"   ✗ FFmpeg Error:\n{result.stderr}")
            return False
    except FileNotFoundError:
        print("   ✗ Error: FFmpeg not found.")
        return False

def main():
    if not os.path.exists(CLIP_DIR):
        os.makedirs(CLIP_DIR)
        print(f"Created directory: {CLIP_DIR}")

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"Starting batch processing from {INPUT_CSV}...\n")
        
        for row in reader:
            video_name = row['video_name'].strip()
            start_str = row['start'].strip()
            end_str = row['end'].strip()
            new_name = row['new_name'].strip()

            if not Path(new_name).suffix:
                new_name += DEFAULT_EXTENSION

            source_path = get_video_path(video_name, VIDEO_DIR)
            output_path = Path(CLIP_DIR) / new_name

            if source_path:
                cut_video(source_path, output_path, start_str, end_str)
            else:
                print(f"❌ Error: Source video '{video_name}' not found")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()