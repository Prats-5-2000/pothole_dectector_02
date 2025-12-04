#!/usr/bin/env python3
"""
Find Newest YOLO Weights

Searches runs/detect/*/weights/ for the most recent best.pt file.

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import sys
from pathlib import Path


def find_newest_weights(runs_dir: Path = Path("runs/detect")) -> Path:
    """
    Find the newest best.pt weights file.
    
    Args:
        runs_dir: Base directory for training runs
        
    Returns:
        Path to newest best.pt file
    """
    if not runs_dir.exists():
        print(f"ERROR: Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all best.pt files
    weight_files = list(runs_dir.glob("*/weights/best.pt"))
    
    if not weight_files:
        print(f"ERROR: No weights found in {runs_dir}/*/weights/best.pt", file=sys.stderr)
        sys.exit(1)
    
    # Sort by modification time, newest first
    weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    newest = weight_files[0]
    return newest


def main():
    parser = argparse.ArgumentParser(
        description="Find newest YOLO weights in runs directory"
    )
    
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=Path('runs/detect'),
        help='Base runs directory (default: runs/detect)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all found weights'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        weight_files = list(args.runs_dir.glob("*/weights/best.pt"))
        weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        print(f"Found {len(weight_files)} weight files:", file=sys.stderr)
        for i, wf in enumerate(weight_files, 1):
            mtime = wf.stat().st_mtime
            print(f"  {i}. {wf} (modified: {mtime})", file=sys.stderr)
        print("", file=sys.stderr)
    
    newest = find_newest_weights(args.runs_dir)
    print(str(newest))  # Output path to stdout
    
    if args.verbose:
        print(f"Selected: {newest}", file=sys.stderr)


if __name__ == '__main__':
    main()
