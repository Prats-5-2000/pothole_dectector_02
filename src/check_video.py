#!/usr/bin/env python3
"""
Video Sanity Check and Re-encoding Utility

Validates video with OpenCV and re-encodes if necessary.

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import cv2


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_video_opencv(video_path: Path) -> bool:
    """
    Check if OpenCV can open and read the video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video can be read, False otherwise
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except Exception as e:
        logger.error(f"OpenCV error: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def reencode_video(input_path: Path, output_path: Path) -> bool:
    """
    Re-encode video using ffmpeg for better compatibility.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        
    Returns:
        True if successful, False otherwise
    """
    if not check_ffmpeg_available():
        logger.error("ffmpeg not found. Install with: brew install ffmpeg")
        return False
    
    logger.info(f"Re-encoding video: {input_path.name}")
    
    # Re-encode with H.264 codec, compatible settings
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',  # Overwrite output
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Re-encoded video saved to: {output_path}")
            return True
        else:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Re-encoding error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check video compatibility and re-encode if needed"
    )
    
    parser.add_argument(
        '--video',
        type=Path,
        required=True,
        help='Path to input video'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Path for re-encoded video (optional)'
    )
    
    parser.add_argument(
        '--force-reencode',
        action='store_true',
        help='Force re-encoding even if video is readable'
    )
    
    args = parser.parse_args()
    
    if not args.video.exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)
    
    # Check if video is readable
    logger.info(f"Checking video: {args.video.name}")
    is_readable = check_video_opencv(args.video)
    
    if is_readable and not args.force_reencode:
        logger.info("Video is readable by OpenCV")
        print(str(args.video))  # Output original path
        sys.exit(0)
    else:
        if not is_readable:
            logger.warning("Video cannot be read by OpenCV, re-encoding required")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.video.parent / f"{args.video.stem}_reencoded.mp4"
        
        # Re-encode
        success = reencode_video(args.video, output_path)
        
        if success:
            # Verify re-encoded video
            if check_video_opencv(output_path):
                logger.info("Re-encoded video verified")
                print(str(output_path))  # Output new path
                sys.exit(0)
            else:
                logger.error("Re-encoded video still not readable")
                sys.exit(1)
        else:
            logger.error("Re-encoding failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
