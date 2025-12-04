#!/usr/bin/env python3
"""
Auto Fine-tuning Script for Pothole Detection

Implements iterative fine-tuning with pseudo-labeling:
1. Run inference on unlabeled data
2. Filter high-confidence predictions as pseudo-labels
3. Fine-tune model on pseudo-labeled data
4. Evaluate improvement and keep best model

Safety features:
- Limited iterations (max 2)
- Backup previous weights
- Abort if no improvement or metrics degrade
- Conservative confidence thresholds

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def backup_weights(weights_path: Path, backup_dir: Path, iteration: int) -> Path:
    """
    Create backup of weights before fine-tuning.
    
    Args:
        weights_path: Path to weights to backup
        backup_dir: Backup directory
        iteration: Iteration number
        
    Returns:
        Path to backup file
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{weights_path.stem}_iter{iteration}_backup.pt"
    shutil.copy(weights_path, backup_path)
    logger.info(f"Weights backed up to: {backup_path}")
    return backup_path


def generate_pseudo_labels(
    model: YOLO,
    image_dir: Path,
    output_labels_dir: Path,
    conf_threshold: float = 0.5,
    min_box_area: int = 200,
    max_images: Optional[int] = None
) -> int:
    """
    Generate pseudo-labels from high-confidence predictions.
    
    Args:
        model: Trained YOLO model
        image_dir: Directory containing images
        output_labels_dir: Output directory for pseudo-labels
        conf_threshold: Minimum confidence for pseudo-labels
        min_box_area: Minimum box area in pixels
        max_images: Maximum number of images to process
        
    Returns:
        Number of images with pseudo-labels
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if max_images:
        images = images[:max_images]
    
    pseudo_labeled_count = 0
    
    for img_path in tqdm(images, desc="Generating pseudo-labels"):
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Extract high-confidence detections
        pseudo_boxes = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                img_h, img_w = result.orig_shape
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter by area
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_box_area:
                        continue
                    
                    # Convert to YOLO format (normalized)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    pseudo_boxes.append([cls, x_center, y_center, width, height])
        
        # Save pseudo-labels
        if len(pseudo_boxes) > 0:
            label_path = output_labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for box in pseudo_boxes:
                    line = ' '.join([f"{x:.6f}" for x in box])
                    f.write(line + '\n')
            pseudo_labeled_count += 1
    
    logger.info(f"Generated pseudo-labels for {pseudo_labeled_count}/{len(images)} images")
    return pseudo_labeled_count


def evaluate_model_on_samples(
    model: YOLO,
    test_images: List[Path],
    conf_threshold: float = 0.25
) -> Dict:
    """
    Simple evaluation on sample images.
    
    Args:
        model: YOLO model
        test_images: List of test image paths
        conf_threshold: Confidence threshold
        
    Returns:
        Dictionary with metrics
    """
    total_detections = 0
    avg_confidence = []
    box_sizes = []
    
    for img_path in test_images:
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    total_detections += 1
                    avg_confidence.append(conf)
                    box_area = (x2 - x1) * (y2 - y1)
                    box_sizes.append(box_area)
    
    metrics = {
        'total_detections': total_detections,
        'avg_confidence': np.mean(avg_confidence) if avg_confidence else 0.0,
        'avg_box_area': np.mean(box_sizes) if box_sizes else 0.0,
        'min_box_area': np.min(box_sizes) if box_sizes else 0.0
    }
    
    return metrics


def auto_finetune(
    initial_weights: Path,
    unlabeled_images_dir: Path,
    test_images_dir: Path,
    output_dir: Path,
    backup_dir: Path,
    max_iterations: int = 2,
    pseudo_conf: float = 0.5,
    min_box_area: int = 200,
    finetune_epochs: int = 5,
    finetune_batch: int = 8,
    device: str = 'auto'
) -> Path:
    """
    Run auto fine-tuning with pseudo-labeling.
    
    Args:
        initial_weights: Path to initial trained weights
        unlabeled_images_dir: Directory with unlabeled images
        test_images_dir: Directory with test images for evaluation
        output_dir: Output directory for fine-tuned models
        backup_dir: Directory for weight backups
        max_iterations: Maximum fine-tuning iterations
        pseudo_conf: Confidence threshold for pseudo-labels
        min_box_area: Minimum box area for pseudo-labels
        finetune_epochs: Epochs per fine-tune iteration
        finetune_batch: Batch size for fine-tuning
        device: Device to use
        
    Returns:
        Path to best weights
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load initial model
    logger.info(f"Loading initial model: {initial_weights}")
    current_weights = initial_weights
    
    # Get test images for evaluation
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if len(test_images) == 0:
        logger.warning("No test images found, skipping evaluation")
        test_images = None
    else:
        test_images = test_images[:50]  # Use first 50 for quick eval
    
    # Baseline evaluation
    if test_images:
        model = YOLO(str(current_weights))
        baseline_metrics = evaluate_model_on_samples(model, test_images)
        logger.info(f"Baseline metrics: {baseline_metrics}")
    else:
        baseline_metrics = None
    
    best_weights = current_weights
    best_score = baseline_metrics['total_detections'] if baseline_metrics else 0
    
    # Iterative fine-tuning
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fine-tuning Iteration {iteration}/{max_iterations}")
        logger.info(f"{'='*50}")
        
        # Backup current weights
        backup_path = backup_weights(current_weights, backup_dir, iteration)
        
        # Load model
        model = YOLO(str(current_weights))
        
        # Generate pseudo-labels
        pseudo_labels_dir = output_dir / f"iter_{iteration}" / "labels"
        pseudo_images_dir = output_dir / f"iter_{iteration}" / "images"
        pseudo_images_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating pseudo-labels...")
        num_pseudo = generate_pseudo_labels(
            model=model,
            image_dir=unlabeled_images_dir,
            output_labels_dir=pseudo_labels_dir,
            conf_threshold=pseudo_conf,
            min_box_area=min_box_area,
            max_images=200  # Limit for safety
        )
        
        if num_pseudo < 10:
            logger.warning(f"Too few pseudo-labels ({num_pseudo}), aborting iteration")
            logger.info(f"Restoring weights from: {backup_path}")
            return best_weights
        
        # Copy images to finetune dataset
        for label_file in pseudo_labels_dir.glob("*.txt"):
            img_name = label_file.stem
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = unlabeled_images_dir / f"{img_name}{ext}"
                if src_img.exists():
                    shutil.copy(src_img, pseudo_images_dir / src_img.name)
                    break
        
        # Create mini dataset YAML
        finetune_yaml = output_dir / f"iter_{iteration}" / "finetune_data.yaml"
        finetune_yaml.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        finetune_config = {
            'path': str((output_dir / f"iter_{iteration}").absolute()),
            'train': 'images',
            'val': 'images',
            'nc': 1,
            'names': ['pothole']
        }
        
        with open(finetune_yaml, 'w') as f:
            yaml.dump(finetune_config, f, default_flow_style=False)
        
        # Fine-tune
        logger.info(f"Fine-tuning on {num_pseudo} pseudo-labeled images...")
        try:
            results = model.train(
                data=str(finetune_yaml),
                epochs=finetune_epochs,
                imgsz=640,
                batch=finetune_batch,
                device=device,
                project=str(output_dir),
                name=f"finetune_iter_{iteration}",
                pretrained=True,
                verbose=False,
                exist_ok=True,
                patience=100,  # Disable early stopping for short runs
                save=True
            )
            
            # Get fine-tuned weights
            finetuned_weights = output_dir / f"finetune_iter_{iteration}" / "weights" / "best.pt"
            
            if not finetuned_weights.exists():
                logger.warning("Fine-tuned weights not found, using backup")
                current_weights = backup_path
                continue
            
            # Evaluate
            if test_images:
                logger.info("Evaluating fine-tuned model...")
                finetuned_model = YOLO(str(finetuned_weights))
                new_metrics = evaluate_model_on_samples(finetuned_model, test_images)
                logger.info(f"New metrics: {new_metrics}")
                
                # Check improvement
                new_score = new_metrics['total_detections']
                
                # Safety check: abort if too many tiny boxes
                if new_metrics['min_box_area'] < 50 and new_metrics['total_detections'] > best_score * 1.5:
                    logger.warning("Detected explosion of tiny boxes, aborting")
                    logger.info(f"Restoring weights from: {backup_path}")
                    break
                
                # Check if improved
                if new_score > best_score:
                    improvement = ((new_score - best_score) / best_score) * 100
                    logger.info(f"Improvement: {improvement:.1f}% (detections: {best_score} -> {new_score})")
                    best_weights = finetuned_weights
                    best_score = new_score
                    current_weights = finetuned_weights
                else:
                    logger.info(f"No improvement (detections: {best_score} -> {new_score}), stopping")
                    break
            else:
                # No evaluation, just use new weights
                current_weights = finetuned_weights
                best_weights = finetuned_weights
        
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            logger.info(f"Restoring weights from: {backup_path}")
            break
    
    logger.info(f"\nAuto fine-tuning complete. Best weights: {best_weights}")
    return best_weights


def main():
    parser = argparse.ArgumentParser(
        description="Auto fine-tune pothole detection model with pseudo-labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic auto fine-tuning
  python src/auto_finetune.py --weights runs/detect/video_merged/weights/best.pt --unlabeled datasets/samples/

  # Custom settings
  python src/auto_finetune.py --weights best.pt --unlabeled data/raw/ --iterations 2 --pseudo-conf 0.6

  # With test evaluation
  python src/auto_finetune.py --weights best.pt --unlabeled data/raw/ --test-images datasets/merged/images/test/
        """
    )
    
    parser.add_argument(
        '--weights',
        type=Path,
        required=True,
        help='Path to initial trained weights'
    )
    
    parser.add_argument(
        '--unlabeled',
        type=Path,
        required=True,
        help='Directory with unlabeled images for pseudo-labeling'
    )
    
    parser.add_argument(
        '--test-images',
        type=Path,
        default=Path('datasets/merged/images/test'),
        help='Directory with test images for evaluation'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('datasets/finetune'),
        help='Output directory (default: datasets/finetune)'
    )
    
    parser.add_argument(
        '--backup',
        type=Path,
        default=Path('backups'),
        help='Backup directory (default: backups/)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=2,
        help='Maximum fine-tuning iterations (default: 2)'
    )
    
    parser.add_argument(
        '--pseudo-conf',
        type=float,
        default=0.5,
        help='Confidence threshold for pseudo-labels (default: 0.5)'
    )
    
    parser.add_argument(
        '--min-box-area',
        type=int,
        default=200,
        help='Minimum box area for pseudo-labels (default: 200)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Epochs per fine-tuning iteration (default: 5)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=8,
        help='Batch size for fine-tuning (default: 8)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.weights.exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)
    
    if not args.unlabeled.exists():
        logger.error(f"Unlabeled images directory not found: {args.unlabeled}")
        sys.exit(1)
    
    # Run auto fine-tuning
    best_weights = auto_finetune(
        initial_weights=args.weights,
        unlabeled_images_dir=args.unlabeled,
        test_images_dir=args.test_images,
        output_dir=args.output,
        backup_dir=args.backup,
        max_iterations=args.iterations,
        pseudo_conf=args.pseudo_conf,
        min_box_area=args.min_box_area,
        finetune_epochs=args.epochs,
        finetune_batch=args.batch,
        device=args.device
    )
    
    logger.info("\n" + "="*50)
    logger.info("Auto fine-tuning pipeline completed!")
    logger.info("="*50)
    logger.info(f"\nBest weights: {best_weights}")
    logger.info(f"\nRun demo with fine-tuned model:")
    logger.info(f"  python src/run_demo.py --weights {best_weights} --video datasets/samples/my_video.mp4")


if __name__ == '__main__':
    main()
