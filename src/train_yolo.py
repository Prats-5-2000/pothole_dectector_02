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
    workers: int = 8,
    verbose: bool = True,
    augment: bool = True,
    copy_paste: float = 0.1,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    erasing: float = 0.4
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
    if augment:
        logger.info("Using extensive augmentations: brightness, blur, perspective, rotation, scale, cutout")
    
    # Setup live training monitor with TensorBoard
    from ultralytics.utils.callbacks.base import add_integration_callbacks
    from torch.utils.tensorboard import SummaryWriter
    
    # Initialize TensorBoard writer
    tb_dir = Path(project) / "tensorboard" / name
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logging to: {tb_dir}")
    logger.info(f"Start TensorBoard with: tensorboard --logdir {project}/tensorboard --bind_all")
    
    # Checkpoint directory
    ckpt_dir = Path(project) / name / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(trainer):
        """Enhanced callback: live metrics, TensorBoard logging, and checkpoint saving"""
        epoch = trainer.epoch + 1
        epochs_total = trainer.epochs
        metrics = trainer.metrics
        
        # Extract key metrics
        box_loss = metrics.get('train/box_loss', 0)
        cls_loss = metrics.get('train/cls_loss', 0)
        dfl_loss = metrics.get('train/dfl_loss', 0)
        
        # Validation metrics
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        
        # Estimate ETA (simple based on epoch time)
        elapsed_time = getattr(trainer, 'train_time_elapsed', 0)
        epochs_remaining = epochs_total - epoch
        eta_seconds = (elapsed_time / epoch * epochs_remaining) if epoch > 0 else 0
        eta_mins = int(eta_seconds / 60)
        
        # Print live update
        print(f"\n{'='*70}")
        print(f"[EPOCH {epoch}/{epochs_total}] TRAINING PROGRESS")
        print(f"{'='*70}")
        print(f"  Train Losses:")
        print(f"    Box Loss:  {box_loss:.4f}")
        print(f"    Cls Loss:  {cls_loss:.4f}")
        print(f"    DFL Loss:  {dfl_loss:.4f}")
        if map50 > 0:  # Only show if validation ran
            print(f"  Validation Metrics:")
            print(f"    mAP@0.5:      {map50:.4f}")
            print(f"    mAP@0.5:0.95: {map50_95:.4f}")
            print(f"    Precision:    {precision:.4f}")
            print(f"    Recall:       {recall:.4f}")
        if eta_mins > 0:
            print(f"  ETA: ~{eta_mins} minutes")
        print(f"{'='*70}\n")
        
        # Log to file
        log_file = Path("logs/training_yolov8m_det.log")
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"[EPOCH {epoch}/{epochs_total}] ")
            f.write(f"Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f}")
            if map50 > 0:
                f.write(f", mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map50_95:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            f.write("\n")
        
        # TensorBoard logging
        tb_writer.add_scalar('Loss/box_loss', box_loss, epoch)
        tb_writer.add_scalar('Loss/cls_loss', cls_loss, epoch)
        tb_writer.add_scalar('Loss/dfl_loss', dfl_loss, epoch)
        if map50 > 0:
            tb_writer.add_scalar('Metrics/mAP@0.5', map50, epoch)
            tb_writer.add_scalar('Metrics/mAP@0.5:0.95', map50_95, epoch)
            tb_writer.add_scalar('Metrics/precision', precision, epoch)
            tb_writer.add_scalar('Metrics/recall', recall, epoch)
        tb_writer.flush()
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
            try:
                import shutil
                last_ckpt = ckpt_dir / "last.pt"
                if last_ckpt.exists():
                    shutil.copy2(last_ckpt, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
    
    # Detect AMP support
    amp_supported = False
    if device in ['cuda', 'mps']:
        try:
            # Test if autocast is supported
            if device == 'cuda':
                from torch.cuda.amp import autocast, GradScaler
                amp_supported = True
                logger.info("AMP (Automatic Mixed Precision) enabled for CUDA")
            elif device == 'mps':
                # MPS AMP support is limited, check PyTorch version
                import torch
                if hasattr(torch, 'autocast') and torch.__version__ >= '2.0':
                    amp_supported = True
                    logger.info("AMP enabled for MPS (PyTorch 2.0+)")
                else:
                    logger.info("MPS detected but AMP not fully supported, using FP32")
        except Exception as e:
            logger.warning(f"AMP check failed: {e}, using FP32")
    
    try:
        # Build training arguments
        train_args = dict(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            pretrained=pretrained,
            patience=patience,
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
            close_mosaic=10,  # Disable mosaic last N epochs
            amp=amp_supported  # Enable AMP if supported
        )
        
        # Add extensive augmentations if enabled
        if augment:
            train_args.update({
                # Extensive augmentation parameters
                'hsv_h': 0.015,      # HSV-Hue augmentation (fraction)
                'hsv_s': 0.7,        # HSV-Saturation augmentation (fraction)
                'hsv_v': 0.4,        # HSV-Value augmentation (fraction) - brightness
                'degrees': 10.0,     # Image rotation ±10 degrees
                'translate': 0.1,    # Image translation (fraction)
                'scale': 0.2,        # Image scale ±20%
                'shear': 0.0,        # Image shear (degrees)
                'perspective': 0.001,# Image perspective warp
                'flipud': 0.0,       # Image flip up-down probability
                'fliplr': 0.5,       # Image flip left-right probability
                'mosaic': mosaic,       # Mosaic augmentation probability
                'mixup': mixup,        # Mixup augmentation probability
                'copy_paste': copy_paste,   # Copy-paste augmentation probability
                'erasing': erasing,      # Random erasing probability (cutout)
                'crop_fraction': 1.0 # Crop fraction
            })
        
        # Register callback for live monitoring
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        
        results = model.train(**train_args)
        
        # Close TensorBoard writer
        tb_writer.close()
        logger.info("TensorBoard writer closed")
        
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
        default='yolov8m.pt',
        help='YOLO model variant (default: yolov8m.pt). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x'
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
        default=30,
        help='Number of training epochs (default: 30)'
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
        default='runs/train',
        help='Project directory (default: runs/train)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='yolov8m_det',
        help='Experiment name (default: yolov8m_det)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
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
    
    parser.add_argument(
        '--copy_paste',
        type=float,
        default=0.1,
        help='Copy-paste augmentation probability (default: 0.1)'
    )
    
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability (default: 1.0)'
    )
    
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.1,
        help='Mixup augmentation probability (default: 0.1)'
    )
    
    parser.add_argument(
        '--erasing',
        type=float,
        default=0.4,
        help='Random erasing probability (default: 0.4)'
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
        verbose=args.verbose,
        copy_paste=args.copy_paste,
        mosaic=args.mosaic,
        mixup=args.mixup,
        erasing=args.erasing
    )
    
    logger.info("\n" + "="*50)
    logger.info("Training pipeline completed!")
    logger.info("="*50)
    logger.info(f"\nNext step: Run demo inference using:")
    logger.info(f"  python src/run_demo.py --weights {best_weights} --video datasets/samples/my_road_video.mp4")


if __name__ == '__main__':
    main()
