#!/usr/bin/env python3
"""
Video 180-Degree Flip Script
Flips a video file 180 degrees (horizontal + vertical flip) using FFmpeg.

Usage:
  python prep_flip_video.py input_video.mp4 [output_video.mp4]
  python prep_flip_video.py video/wahoo/2026-03-11_030438.mp4 video/2026-03-11_030438_flip.mp4
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def flip_video_180(input_path, output_path=None):
    """
    Flip video 180 degrees using FFmpeg.

    Args:
        input_path: Path to input video file
        output_path: Path for output video (optional)
    """

    # Validate input file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return False

    # Generate output filename if not provided
    input_file = Path(input_path)
    if output_path is None:
        output_path = f"{input_file.stem}_flipped180{input_file.suffix}"
    
    # Ensure output is in the root folder (current working directory)
    # unless a path was explicitly provided.
    output_path = Path(output_path).name

    print(f"Flipping '{input_path}' 180 degrees...")
    print(f"Output will be saved as: {output_path}")

    # Build FFmpeg command
    # 'hflip,vflip' effectively rotates the video 180 degrees
    cmd = [
        'ffmpeg', 
        '-i', str(input_path),
        '-vf', 'hflip,vflip',
        '-c:a', 'copy', # Copy audio without re-encoding
        '-y',           # Overwrite output file if it exists
        str(output_path)
    ]

    try:
        # Run the FFmpeg command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Print output to console
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\nSuccessfully flipped video: {output_path}")
            return True
        else:
            print(f"\nFFmpeg process failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Flip a video 180 degrees.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", nargs="?", help="Optional: Path to the output video file (default: input_flipped180.mp4 in root)")

    args = parser.parse_args()

    success = flip_video_180(args.input, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
