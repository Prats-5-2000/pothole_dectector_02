#!/usr/bin/env python3
"""
Convert YOLO bounding box dataset to COCO segmentation format.

This script walks through YOLO-format labels (class x_center y_center width height)
and converts each bounding box into a rectangular polygon segmentation mask.

Output:
  - annotations/instances_rectangles.json (COCO format with polygon annotations)
  - Optional: masks/ directory with binary mask PNGs

Usage:
  python scripts/convert_bbox_to_coco_seg.py \
    --images datasets/merged/images/train \
    --labels datasets/merged/labels/train \
    --output annotations/train_coco_seg.json \
    --classes-file configs/classes.txt

Note: This creates rectangular polygons from bounding boxes. For true segmentation,
      you would need instance masks from a segmentation annotation tool.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np


def yolo_to_pixel_bbox(x_center: float, y_center: float, width: float, height: float,
                       img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates to pixel bounding box.
    
    Args:
        x_center, y_center, width, height: YOLO normalized coords (0-1)
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        (x_min, y_min, x_max, y_max) in pixels
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return x_min, y_min, x_max, y_max


def bbox_to_polygon(x_min: int, y_min: int, x_max: int, y_max: int) -> List[float]:
    """
    Convert bounding box to COCO polygon format (rectangular).
    
    Returns:
        Flat list [x1,y1, x2,y2, x3,y3, x4,y4] for rectangle vertices
    """
    return [
        x_min, y_min,  # top-left
        x_max, y_min,  # top-right
        x_max, y_max,  # bottom-right
        x_min, y_max   # bottom-left
    ]


def compute_area(x_min: int, y_min: int, x_max: int, y_max: int) -> float:
    """Compute bounding box area."""
    return float((x_max - x_min) * (y_max - y_min))


def load_class_names(classes_file: Path) -> Dict[int, str]:
    """
    Load class names from file (one per line).
    
    Returns:
        Dict mapping class_id to class_name
    """
    if not classes_file.exists():
        print(f"Warning: Classes file not found: {classes_file}")
        return {}
    
    class_names = {}
    with open(classes_file, 'r') as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                class_names[idx] = name
    
    return class_names


def convert_dataset(images_dir: Path, labels_dir: Path, output_json: Path,
                   classes_file: Path = None, generate_masks: bool = False,
                   masks_dir: Path = None) -> None:
    """
    Convert YOLO bbox dataset to COCO segmentation format.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO .txt label files
        output_json: Output COCO JSON file path
        classes_file: Optional file with class names (one per line)
        generate_masks: If True, generate binary mask PNGs
        masks_dir: Directory to save masks (if generate_masks=True)
    """
    
    # Load class names
    class_names = {}
    if classes_file and classes_file.exists():
        class_names = load_class_names(classes_file)
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Converted from YOLO bbox to COCO segmentation (rectangular polygons)",
            "version": "1.0",
            "year": 2025,
            "date_created": "2025-12-04"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Collect all unique class IDs from labels
    all_class_ids = set()
    label_files = list(labels_dir.glob("*.txt"))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    all_class_ids.add(class_id)
    
    # Create categories
    for class_id in sorted(all_class_ids):
        category_name = class_names.get(class_id, f"class_{class_id}")
        coco_data["categories"].append({
            "id": class_id + 1,  # COCO uses 1-indexed categories
            "name": category_name,
            "supercategory": "object"
        })
    
    print(f"Found {len(all_class_ids)} classes: {sorted(all_class_ids)}")
    
    # Process each image
    image_id = 1
    annotation_id = 1
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    for image_path in sorted(image_files):
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_width,
            "height": img_height
        })
        
        # Process labels if they exist
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to pixel bbox
                        x_min, y_min, x_max, y_max = yolo_to_pixel_bbox(
                            x_center, y_center, width, height, img_width, img_height
                        )
                        
                        # Clamp to image bounds
                        x_min = max(0, min(x_min, img_width - 1))
                        y_min = max(0, min(y_min, img_height - 1))
                        x_max = max(0, min(x_max, img_width))
                        y_max = max(0, min(y_max, img_height))
                        
                        # Skip invalid boxes
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        # Create polygon (rectangle)
                        polygon = bbox_to_polygon(x_min, y_min, x_max, y_max)
                        area = compute_area(x_min, y_min, x_max, y_max)
                        
                        # Add annotation
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # COCO 1-indexed
                            "segmentation": [polygon],  # List of polygons
                            "area": area,
                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],  # [x,y,w,h]
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line in {label_path}: {line.strip()} - {e}")
                        continue
        
        image_id += 1
        
        if image_id % 100 == 0:
            print(f"Processed {image_id - 1} images, {annotation_id - 1} annotations...")
    
    # Save COCO JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_json}")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")
    
    # Optionally generate mask images
    if generate_masks and masks_dir:
        print(f"\nGenerating mask images to {masks_dir}...")
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        for img_data in coco_data["images"][:10]:  # Limit to first 10 for demo
            img_id = img_data["id"]
            img_width = img_data["width"]
            img_height = img_data["height"]
            
            # Create blank mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Draw all annotations for this image
            for ann in coco_data["annotations"]:
                if ann["image_id"] == img_id:
                    x_min, y_min, w, h = ann["bbox"]
                    x_max = x_min + w
                    y_max = y_min + h
                    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255
            
            # Save mask
            mask_path = masks_dir / f"{img_data['file_name'].rsplit('.', 1)[0]}_mask.png"
            Image.fromarray(mask).save(mask_path)
        
        print(f"   Generated {min(10, len(coco_data['images']))} sample masks")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO bbox dataset to COCO segmentation format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--images", "-i",
        type=Path,
        required=True,
        help="Path to images directory"
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=Path,
        required=True,
        help="Path to YOLO labels directory (*.txt files)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output COCO JSON file path"
    )
    
    parser.add_argument(
        "--classes-file", "-c",
        type=Path,
        default=None,
        help="Optional file with class names (one per line)"
    )
    
    parser.add_argument(
        "--generate-masks",
        action="store_true",
        help="Generate binary mask PNG files (demo only, first 10 images)"
    )
    
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("masks"),
        help="Directory to save mask images (if --generate-masks)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.images.exists():
        print(f"Error: Images directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)
    
    if not args.labels.exists():
        print(f"Error: Labels directory not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    
    # Run conversion
    convert_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_json=args.output,
        classes_file=args.classes_file,
        generate_masks=args.generate_masks,
        masks_dir=args.masks_dir if args.generate_masks else None
    )


if __name__ == "__main__":
    main()
