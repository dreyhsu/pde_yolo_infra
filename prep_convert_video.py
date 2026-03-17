#!/usr/bin/env python3
"""
Video to H.264 Converter Script
Converts video files to H.264 codec with various quality and resolution options.

Usage:
  python convert_to_h264.py input_video.mp4 [output_video.mp4]
  python convert_to_h264.py input_video.mp4 --quality high [output_video.mp4]
  python convert_to_h264.py input_video.mp4 --crf 23 --preset medium [output_video.mp4]
  python convert_to_h264.py input_video.mp4 --resize 1920 1080 [output_video.mp4]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def convert_to_h264(input_path, output_path=None, crf=23, preset='medium',
                    resize_width=None, resize_height=None, audio_codec='aac'):
    """
    Convert video to H.264 format with specified quality settings

    Args:
        input_path: Path to input video file
        output_path: Path for output video (optional)
        crf: Constant Rate Factor (0-51, lower is better quality, default 23)
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        resize_width: Width for output video (optional)
        resize_height: Height for output video (optional)
        audio_codec: Audio codec to use (aac, copy, or none)
    """

    # Validate input file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return False

    # Generate output filename if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = f"{input_file.stem}_h264{input_file.suffix}"

    # Build FFmpeg command
    cmd = ['ffmpeg', '-i', input_path]

    # Add resize filter if dimensions provided
    if resize_width is not None and resize_height is not None:
        resize_filter = f'scale={resize_width}:{resize_height}:force_original_aspect_ratio=disable,setsar=1'
        cmd.extend(['-vf', resize_filter])

    # Add video codec and settings
    # Use simple quality-based bitrate since libx264 with CRF/preset may not be available
    # Calculate bitrate based on CRF (approximate conversion)
    if crf <= 18:
        video_bitrate = '5M'  # High quality
    elif crf <= 23:
        video_bitrate = '3M'  # Medium quality
    elif crf <= 28:
        video_bitrate = '1.5M'  # Lower quality
    else:
        video_bitrate = '1M'  # Low quality

    cmd.extend([
        '-c:v', 'h264_mf',      # Use H.264 codec via MediaFoundation
        '-b:v', video_bitrate,  # Video bitrate
        '-pix_fmt', 'yuv420p'   # Pixel format for compatibility
    ])

    # Add audio codec settings
    if audio_codec == 'copy':
        cmd.extend(['-c:a', 'copy'])
    elif audio_codec == 'aac':
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
    elif audio_codec == 'none':
        cmd.extend(['-an'])  # No audio
    else:
        cmd.extend(['-c:a', audio_codec])

    # Overwrite output file if exists
    cmd.extend(['-y', output_path])

    print(f"Converting video to H.264:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Video bitrate: {video_bitrate}")
    if resize_width and resize_height:
        print(f"  Resize to: {resize_width}x{resize_height}")
    print(f"  Audio codec: {audio_codec}")
    print()

    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Video converted successfully: {output_path}")

            # Show file sizes
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Input size: {input_size:.2f} MB")
            print(f"  Output size: {output_size:.2f} MB")
            print(f"  Compression ratio: {(output_size/input_size)*100:.1f}%")
            return True
        else:
            print(f"✗ Error converting video:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to H.264 format using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with default settings
  python convert_to_h264.py input.mp4
  python convert_to_h264.py input.mp4 output_h264.mp4

  # High quality conversion (lower CRF = better quality)
  python convert_to_h264.py input.mp4 --crf 18 --preset slow

  # Fast conversion with lower quality
  python convert_to_h264.py input.mp4 --crf 28 --preset veryfast

  # Convert and resize
  python convert_to_h264.py input.mp4 --resize 1920 1080

  # Use quality presets
  python convert_to_h264.py input.mp4 --quality high
  python convert_to_h264.py input.mp4 --quality medium
  python convert_to_h264.py input.mp4 --quality low

  # Copy audio without re-encoding
  python convert_to_h264.py input.mp4 --audio copy

  # Remove audio
  python convert_to_h264.py input.mp4 --audio none

Quality Guide:
  CRF values (0-51, default 23):
    - 18-22: High quality (larger file size)
    - 23-28: Medium quality (balanced)
    - 29-35: Low quality (smaller file size)

  Presets (speed vs compression):
    - ultrafast, superfast, veryfast: Fast encoding, larger files
    - faster, fast, medium: Balanced
    - slow, slower, veryslow: Slow encoding, smaller files
        """
    )

    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', nargs='?', help='Path to output video file (optional)')

    # Quality settings
    parser.add_argument('--crf', type=int, default=23,
                       help='Constant Rate Factor (0-51, lower is better quality, default: 23)')
    parser.add_argument('--preset', default='medium',
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                               'medium', 'slow', 'slower', 'veryslow'],
                       help='Encoding preset (default: medium)')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'],
                       help='Quality preset (overrides --crf and --preset)')

    # Resize options
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize video to specified width and height')

    # Audio options
    parser.add_argument('--audio', default='aac',
                       choices=['aac', 'copy', 'none'],
                       help='Audio codec (default: aac)')

    args = parser.parse_args()

    # Apply quality presets
    crf = args.crf
    preset = args.preset

    if args.quality == 'high':
        crf = 18
        preset = 'slow'
    elif args.quality == 'medium':
        crf = 23
        preset = 'medium'
    elif args.quality == 'low':
        crf = 28
        preset = 'fast'

    # Validate CRF value
    if not 0 <= crf <= 51:
        print("Error: CRF value must be between 0 and 51")
        sys.exit(1)

    # Parse resize dimensions
    resize_width, resize_height = None, None
    if args.resize:
        resize_width, resize_height = args.resize

    success = convert_to_h264(
        args.input_video,
        args.output_video,
        crf=crf,
        preset=preset,
        resize_width=resize_width,
        resize_height=resize_height,
        audio_codec=args.audio
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# Examples:
# Basic conversion:
# python convert_to_h264.py video\r550_SOP.mp4 video\r550_SOP_h264_640.mp4 --resize 640 640

# High quality conversion:
# python convert2h264.py clip\cable_1.mp4 --quality high cable_1_con.mp4

# Custom quality and resize:
# python convert_to_h264.py video\input.mp4 --crf 20 --preset slow --resize 1920 1080 output.mp4

# Fast conversion:
# python convert_to_h264.py video\input.mp4 --crf 28 --preset veryfast
