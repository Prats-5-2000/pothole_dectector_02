#!/usr/bin/env python3
"""
YOLOv8 Training Script for Pothole Detection

Trains YOLOv8 model on prepared dataset with configurable parameters.
Supports CPU/GPU auto-detection and various training modes.

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Auto-detect best available device (GPU/CPU).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Apple Silicon GPU (MPS) detected")
    else:
        device = 'cpu'
        logger.info("No GPU detected, using CPU")
    
    return device


def train_yolo(
    model_name: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    pretrained: bool = True,
    patience: int = 50,
    save_period: int = 10,
    workers: int = 8,
    verbose: bool = True
) -> Path:
    """
    Train YOLOv8 model.
    
    Args:
        model_name: Model variant (e.g., 'yolov8n.pt', 'yolov8s.pt')
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        project: Project directory for runs
        name: Experiment name
        pretrained: Use pretrained weights
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        workers: Number of data loader workers
        verbose: Verbose output
        
    Returns:
        Path to best model weights
    """
    # Auto-detect device if requested
    if device == 'auto':
        device = detect_device()
    
    # Adjust batch size for CPU
    if device == 'cpu' and batch > 8:
        logger.warning(f"Reducing batch size from {batch} to 8 for CPU training")
        batch = 8
    
    # Adjust workers for CPU
    if device == 'cpu' and workers > 4:
        workers = 4
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)
    
    # Verify data YAML exists
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.error("Run data_pipeline.py first to prepare the dataset")
        sys.exit(1)
    
    logger.info("Training configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Project: {project}")
    logger.info(f"  Name: {name}")
    
    # Train
    logger.info("\nStarting training...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            pretrained=pretrained,
            patience=patience,
            save_period=save_period,
            workers=workers,
            verbose=verbose,
            exist_ok=True,
            # Additional useful settings
            save=True,
            plots=True,
            val=True,
            cache=False,  # Don't cache on disk for Mac compatibility
            rect=False,   # No rect training for simplicity
            cos_lr=True,  # Cosine learning rate scheduler
            close_mosaic=10  # Disable mosaic last N epochs
        )
        
        logger.info("\nTraining completed successfully!")
        
        # Get path to best weights
        weights_dir = Path(project) / name / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        
        if best_weights.exists():
            logger.info(f"Best weights saved to: {best_weights}")
        if last_weights.exists():
            logger.info(f"Last weights saved to: {last_weights}")
        
        return best_weights if best_weights.exists() else last_weights
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for pothole detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 epochs)
  python src/train_yolo.py --epochs 5

  # Full training with custom model
  python src/train_yolo.py --model yolov8s.pt --epochs 100 --batch 16

  # Debug mode (very small run)
  python src/train_yolo.py --debug

  # Tiny model for fast testing
  python src/train_yolo.py --tiny --epochs 10

  # Specify device explicitly
  python src/train_yolo.py --device cuda --epochs 50
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model variant (default: yolov8n.pt). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x'
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('configs/dataset_video.yaml'),
        help='Path to dataset YAML file (default: configs/dataset_video.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16, auto-adjusted for CPU)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: auto, cuda, mps, cpu (default: auto)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='video_merged',
        help='Experiment name (default: video_merged)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of data loader workers (default: 8, auto-adjusted for CPU)'
    )
    
    parser.add_argument(
        '--tiny',
        action='store_true',
        help='Use lightest model (yolov8n) with reduced settings for fast testing'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: 2 epochs, small batch, for quick testing'
    )
    
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch without pretrained weights'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Apply tiny mode settings
    if args.tiny:
        args.model = 'yolov8n.pt'
        args.imgsz = 320
        args.batch = 8
        logger.info("Tiny mode enabled: using yolov8n, 320px images, batch=8")
    
    # Apply debug mode settings
    if args.debug:
        args.epochs = 2
        args.batch = 4
        args.imgsz = 320
        args.patience = 100  # Disable early stopping
        args.name = 'debug'
        logger.info("Debug mode enabled: 2 epochs, batch=4, 320px images")
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Train
    best_weights = train_yolo(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        patience=args.patience,
        workers=args.workers,
        verbose=args.verbose
    )
    
    logger.info("\n" + "="*50)
    logger.info("Training pipeline completed!")
    logger.info("="*50)
    logger.info(f"\nNext step: Run demo inference using:")
    logger.info(f"  python src/run_demo.py --weights {best_weights} --video datasets/samples/my_road_video.mp4")


if __name__ == '__main__':
    main()
