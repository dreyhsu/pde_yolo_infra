import os
import subprocess
import argparse
import tempfile
from pathlib import Path

def concat_videos(video1, video2, output_path, force_reencode=False):
    """
    Concatenates two videos.
    If force_reencode is False, it attempts a fast stream copy first.
    If that fails or if force_reencode is True, it uses the 'concat' filter
    which is much more robust for videos with different properties.
    """
    
    # Validate input files
    if not os.path.exists(video1):
        print(f"Error: Video 1 '{video1}' not found")
        return False
    if not os.path.exists(video2):
        print(f"Error: Video 2 '{video2}' not found")
        return False

    # Create a temporary file for the concat list (needed for Attempt 1)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(f"file '{os.path.abspath(video1)}'\n")
        f.write(f"file '{os.path.abspath(video2)}'\n")
        temp_list_path = f.name

    success = False
    try:
        # --- Attempt 1: Stream Copy (Fast) ---
        if not force_reencode:
            print(f"Concatenating (Fast Copy): {video1} + {video2} -> {output_path}")
            cmd_copy = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', temp_list_path,
                '-c', 'copy', '-y', output_path
            ]
            result = subprocess.run(cmd_copy, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Success (fast copy)")
                success = True
            else:
                print("! Fast copy failed. Falling back to robust re-encoding...")

        # --- Attempt 2: Concat Filter (Robust Re-encoding) ---
        if not success:
            print(f"Concatenating (Robust Re-encode): {video1} + {video2} -> {output_path}")
            
            # This filter method is much more reliable for mismatched videos
            # We use anr=0 to ignore audio if it's missing, but here we'll assume 
            # we want to try and include audio if possible.
            # v=1:a=1 means 1 video stream and 1 audio stream output.
            filter_str = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
            
            cmd_reencode = [
                'ffmpeg',
                '-i', video1,
                '-i', video2,
                '-filter_complex', filter_str,
                '-map', '[v]',
                '-map', '[a]',
                '-c:v', 'h264_mf', # Using Windows MediaFoundation for speed
                '-b:v', '5M',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            
            # If the above fails (e.g. one video has no audio), try without audio
            result_re = subprocess.run(cmd_reencode, capture_output=True, text=True)
            
            if result_re.returncode != 0:
                print("! Re-encode with audio failed. Retrying without audio...")
                cmd_no_audio = [
                    'ffmpeg',
                    '-i', video1,
                    '-i', video2,
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
                    '-map', '[v]',
                    '-c:v', 'h264_mf',
                    '-b:v', '5M',
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    '-y',
                    output_path
                ]
                result_re = subprocess.run(cmd_no_audio, capture_output=True, text=True)

            if result_re.returncode == 0:
                print(f"✓ Success (re-encoded)")
                success = True
            else:
                print(f"✗ Error: Concatenation failed")
                print(result_re.stderr)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Concatenate two MP4 videos using FFmpeg")
    parser.add_argument('video1', help='Path to the first video file')
    parser.add_argument('video2', help='Path to the second video file')
    parser.add_argument('-o', '--output', help='Output path (default: concatenated.mp4)', default='concatenated.mp4')
    parser.add_argument('--reencode', action='store_true', help='Force re-encoding (slower but much more robust)')

    args = parser.parse_args()
    
    success = concat_videos(args.video1, args.video2, args.output, force_reencode=args.reencode)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
