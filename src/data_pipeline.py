#!/usr/bin/env python3
"""
Data Pipeline for Pothole Detection Project

This module handles:
- Video frame extraction from datasets/Video_dataset/
- Mask video/image to YOLO detection format conversion
- Pascal VOC XML to YOLO format conversion (AndrewMVD dataset)
- Merged dataset generation for training

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from skimage import measure
from tqdm import tqdm


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float = 5.0) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check disk space
        required_gb: Required space in GB
        
    Returns:
        True if sufficient space available
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < required_gb:
            logger.warning(f"Low disk space: {free_gb:.2f} GB free (recommended: {required_gb} GB)")
            return False
        logger.info(f"Disk space check: {free_gb:.2f} GB available")
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: int = 1,
    prefix: Optional[str] = None
) -> List[Path]:
    """
    Extract frames from a video file at specified fps.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        prefix: Optional prefix for frame names
        
    Returns:
        List of paths to extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        logger.warning(f"Could not determine FPS for {video_path}, using default 30")
        video_fps = 30
    
    frame_interval = max(1, int(video_fps / fps))
    
    if prefix is None:
        prefix = video_path.stem
    
    frame_count = 0
    extracted_count = 0
    saved_frames = []
    
    with tqdm(desc=f"Extracting from {video_path.name}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_name = f"{prefix}_frame_{extracted_count:06d}.jpg"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(frame_path)
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    logger.info(f"Extracted {extracted_count} frames from {video_path.name}")
    return saved_frames


def mask_to_yolo_boxes(
    mask: np.ndarray,
    min_area: int = 100,
    class_id: int = 0
) -> List[List[float]]:
    """
    Convert binary mask to YOLO format bounding boxes using connected components.
    
    Args:
        mask: Binary mask image (H, W)
        min_area: Minimum area threshold for detection
        class_id: Class ID for YOLO format
        
    Returns:
        List of YOLO format boxes [class_id, x_center, y_center, width, height] (normalized)
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)
    
    h, w = mask.shape
    boxes = []
    
    for region in regions:
        if region.area < min_area:
            continue
        
        # Get bounding box
        min_row, min_col, max_row, max_col = region.bbox
        
        # Convert to YOLO format (normalized)
        x_center = ((min_col + max_col) / 2) / w
        y_center = ((min_row + max_row) / 2) / h
        box_width = (max_col - min_col) / w
        box_height = (max_row - min_row) / h
        
        boxes.append([class_id, x_center, y_center, box_width, box_height])
    
    return boxes


def process_mask_video_or_images(
    mask_path: Path,
    output_dir: Path,
    fps: int = 1,
    min_area: int = 100,
    prefix: Optional[str] = None
) -> List[Path]:
    """
    Process mask video or sequence of mask images.
    
    Args:
        mask_path: Path to mask video or directory
        output_dir: Output directory for mask frames
        fps: FPS for video extraction
        min_area: Minimum area for detections
        prefix: Prefix for output files
        
    Returns:
        List of saved mask frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mask_path.is_file():
        # Video file
        return extract_frames_from_video(mask_path, output_dir, fps, prefix)
    else:
        logger.warning(f"Mask path {mask_path} is not a file, skipping")
        return []


def voc_xml_to_yolo(
    xml_path: Path,
    class_id: int = 0
) -> List[List[float]]:
    """
    Convert Pascal VOC XML annotation to YOLO format.
    
    Args:
        xml_path: Path to XML file
        class_id: Class ID for YOLO (default: 0 for pothole)
        
    Returns:
        List of YOLO format boxes
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            boxes.append([class_id, x_center, y_center, box_width, box_height])
        
        return boxes
    except Exception as e:
        logger.error(f"Error parsing XML {xml_path}: {e}")
        return []


def save_yolo_labels(boxes: List[List[float]], output_path: Path):
    """
    Save YOLO format labels to file.
    
    Args:
        boxes: List of YOLO format boxes
        output_path: Output label file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for box in boxes:
            line = ' '.join([f"{x:.6f}" for x in box])
            f.write(line + '\n')


def process_video_dataset_split(
    split_dir: Path,
    output_frames_dir: Path,
    output_masks_dir: Path,
    fps: int = 1,
    min_area: int = 100,
    dry_run: bool = False
) -> Dict[str, List[Path]]:
    """
    Process one split (train/val/test) of video dataset.
    
    Args:
        split_dir: Directory containing rgb/ and mask/ subdirs
        output_frames_dir: Output for RGB frames
        output_masks_dir: Output for mask frames
        fps: Extraction FPS
        min_area: Minimum detection area
        dry_run: If True, process only first video pair
        
    Returns:
        Dictionary with 'frames' and 'masks' lists
    """
    rgb_dir = split_dir / "rgb"
    mask_dir = split_dir / "mask"
    
    if not rgb_dir.exists():
        logger.warning(f"RGB directory not found: {rgb_dir}")
        return {'frames': [], 'masks': []}
    
    results = {'frames': [], 'masks': []}
    
    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    rgb_videos = [f for f in rgb_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if dry_run and len(rgb_videos) > 0:
        rgb_videos = rgb_videos[:1]
        logger.info(f"Dry-run mode: processing only first video")
    
    for rgb_video in rgb_videos:
        logger.info(f"Processing RGB video: {rgb_video.name}")
        
        # Extract RGB frames
        rgb_frames = extract_frames_from_video(
            rgb_video,
            output_frames_dir,
            fps=fps,
            prefix=rgb_video.stem
        )
        results['frames'].extend(rgb_frames)
        
        # Process corresponding mask
        if mask_dir.exists():
            # Try to find corresponding mask video
            mask_video = mask_dir / rgb_video.name
            if mask_video.exists():
                logger.info(f"Processing mask video: {mask_video.name}")
                mask_frames = process_mask_video_or_images(
                    mask_video,
                    output_masks_dir,
                    fps=fps,
                    prefix=rgb_video.stem
                )
                results['masks'].extend(mask_frames)
            else:
                logger.warning(f"No matching mask video found for {rgb_video.name}")
    
    return results


def convert_masks_to_yolo_labels(
    mask_frames: List[Path],
    output_labels_dir: Path,
    min_area: int = 100
):
    """
    Convert mask frames to YOLO label files.
    
    Args:
        mask_frames: List of mask frame paths
        output_labels_dir: Output directory for label files
        min_area: Minimum area threshold
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for mask_path in tqdm(mask_frames, desc="Converting masks to YOLO labels"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Could not read mask: {mask_path}")
            continue
        
        boxes = mask_to_yolo_boxes(mask, min_area=min_area)
        
        label_path = output_labels_dir / f"{mask_path.stem}.txt"
        save_yolo_labels(boxes, label_path)


def process_andrewmvd_dataset(
    andrewmvd_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path
) -> int:
    """
    Process AndrewMVD pothole dataset (Pascal VOC format).
    
    Args:
        andrewmvd_dir: Root directory of AndrewMVD dataset
        output_images_dir: Output for images
        output_labels_dir: Output for labels
        
    Returns:
        Number of processed images
    """
    images_dir = andrewmvd_dir / "images"
    annotations_dir = andrewmvd_dir / "annotations"
    
    if not images_dir.exists() or not annotations_dir.exists():
        logger.warning(f"AndrewMVD dataset incomplete at {andrewmvd_dir}")
        return 0
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(annotations_dir.glob("*.xml"))
    processed = 0
    
    for xml_path in tqdm(xml_files, desc="Processing AndrewMVD dataset"):
        # Find corresponding image
        img_name = xml_path.stem
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            logger.warning(f"No image found for {xml_path.name}")
            continue
        
        # Convert annotation
        boxes = voc_xml_to_yolo(xml_path)
        
        if len(boxes) > 0:
            # Copy image
            shutil.copy(img_path, output_images_dir / img_path.name)
            
            # Save label
            label_path = output_labels_dir / f"{img_path.stem}.txt"
            save_yolo_labels(boxes, label_path)
            processed += 1
    
    logger.info(f"Processed {processed} images from AndrewMVD dataset")
    return processed


def create_merged_dataset(
    video_dataset_dir: Path,
    andrewmvd_dir: Path,
    merged_output_dir: Path,
    fps: int = 1,
    min_area: int = 100,
    dry_run: bool = False
):
    """
    Create merged YOLO dataset from all sources.
    
    Args:
        video_dataset_dir: Path to Video_dataset
        andrewmvd_dir: Path to AndrewMVD dataset
        merged_output_dir: Output directory for merged dataset
        fps: FPS for frame extraction
        min_area: Minimum detection area
        dry_run: Process only first video per split
    """
    logger.info("Starting dataset merge process...")
    
    # Check disk space
    check_disk_space(merged_output_dir.parent, required_gb=5.0)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {split} split")
        logger.info(f"{'='*50}")
        
        split_dir = video_dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        # Setup output directories
        output_images = merged_output_dir / "images" / split
        output_labels = merged_output_dir / "labels" / split
        
        # Temporary directories for frames
        temp_rgb_frames = video_dataset_dir / split / "rgb_frames"
        temp_mask_frames = video_dataset_dir / split / "mask_frames"
        
        # Process videos
        results = process_video_dataset_split(
            split_dir,
            temp_rgb_frames,
            temp_mask_frames,
            fps=fps,
            min_area=min_area,
            dry_run=dry_run
        )
        
        # Convert masks to YOLO labels and copy to merged
        if results['masks']:
            convert_masks_to_yolo_labels(
                results['masks'],
                output_labels,
                min_area=min_area
            )
        
        # Copy RGB frames to merged images
        output_images.mkdir(parents=True, exist_ok=True)
        for frame_path in results['frames']:
            shutil.copy(frame_path, output_images / frame_path.name)
        
        logger.info(f"Split {split}: {len(results['frames'])} images processed")
    
    # Process AndrewMVD dataset (add to train split)
    if andrewmvd_dir.exists():
        logger.info(f"\n{'='*50}")
        logger.info("Processing AndrewMVD dataset")
        logger.info(f"{'='*50}")
        
        process_andrewmvd_dataset(
            andrewmvd_dir,
            merged_output_dir / "images" / "train",
            merged_output_dir / "labels" / "train"
        )
    
    logger.info("\nDataset merge complete!")


def create_dataset_yaml(
    merged_dir: Path,
    output_yaml: Path,
    class_names: List[str] = None
):
    """
    Create YOLO dataset YAML configuration file.
    
    Args:
        merged_dir: Path to merged dataset directory
        output_yaml: Output YAML file path
        class_names: List of class names (default: ['pothole'])
    """
    if class_names is None:
        class_names = ['pothole']
    
    # Count files in each split
    stats = {}
    for split in ['train', 'val', 'test']:
        split_dir = merged_dir / "images" / split
        if split_dir.exists():
            stats[split] = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
        else:
            stats[split] = 0
    
    dataset_config = {
        'path': str(merged_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_yaml, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\nDataset YAML created: {output_yaml}")
    logger.info(f"Dataset statistics:")
    logger.info(f"  Train: {stats['train']} images")
    logger.info(f"  Val:   {stats['val']} images")
    logger.info(f"  Test:  {stats['test']} images")
    logger.info(f"  Total: {sum(stats.values())} images")


def main():
    parser = argparse.ArgumentParser(
        description="Data Pipeline for Pothole Detection - Extract, convert, and merge datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset processing
  python src/data_pipeline.py

  # Dry-run (process only first video per split)
  python src/data_pipeline.py --dry-run

  # Custom extraction settings
  python src/data_pipeline.py --fps 2 --min-area 200

  # Clean temporary files after processing
  python src/data_pipeline.py --clean
        """
    )
    
    parser.add_argument(
        '--video-dataset',
        type=Path,
        default=Path('datasets/Video_dataset'),
        help='Path to Video_dataset directory (default: datasets/Video_dataset)'
    )
    
    parser.add_argument(
        '--andrewmvd',
        type=Path,
        default=Path('datasets/AndrewMVD pothole dataset'),
        help='Path to AndrewMVD dataset (default: datasets/AndrewMVD pothole dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('datasets/merged'),
        help='Output directory for merged dataset (default: datasets/merged)'
    )
    
    parser.add_argument(
        '--config-output',
        type=Path,
        default=Path('configs/dataset_video.yaml'),
        help='Output path for dataset YAML (default: configs/dataset_video.yaml)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to extract from videos (default: 1)'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=100,
        help='Minimum area threshold for detections in pixels (default: 100)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process only first video per split for testing'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean temporary frame directories after processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Log configuration
    logger.info("Data Pipeline Configuration:")
    logger.info(f"  Video dataset: {args.video_dataset}")
    logger.info(f"  AndrewMVD: {args.andrewmvd}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  FPS: {args.fps}")
    logger.info(f"  Min area: {args.min_area}")
    logger.info(f"  Dry-run: {args.dry_run}")
    
    # Check if datasets exist
    if not args.video_dataset.exists():
        logger.error(f"Video dataset not found: {args.video_dataset}")
        sys.exit(1)
    
    # Create merged dataset
    create_merged_dataset(
        args.video_dataset,
        args.andrewmvd,
        args.output,
        fps=args.fps,
        min_area=args.min_area,
        dry_run=args.dry_run
    )
    
    # Create dataset YAML
    create_dataset_yaml(args.output, args.config_output)
    
    # Clean temporary files if requested
    if args.clean:
        logger.info("\nCleaning temporary frame directories...")
        for split in ['train', 'val', 'test']:
            temp_rgb = args.video_dataset / split / "rgb_frames"
            temp_mask = args.video_dataset / split / "mask_frames"
            
            if temp_rgb.exists():
                shutil.rmtree(temp_rgb)
                logger.info(f"Removed {temp_rgb}")
            
            if temp_mask.exists():
                shutil.rmtree(temp_mask)
                logger.info(f"Removed {temp_mask}")
    
    logger.info("\n" + "="*50)
    logger.info("Data pipeline completed successfully!")
    logger.info("="*50)
    logger.info(f"\nNext step: Train model using:")
    logger.info(f"  python src/train_yolo.py --epochs 5")


if __name__ == '__main__':
    main()
