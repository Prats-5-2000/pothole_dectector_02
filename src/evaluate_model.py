#!/usr/bin/env python3
"""
Model Evaluation Script for Pothole Detection

Computes comprehensive metrics: mAP, precision, recall, F1 score, etc.

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from ultralytics import YOLO
import numpy as np


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    weights: Path,
    data_yaml: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Evaluate YOLOv8 model and compute comprehensive metrics.
    
    Args:
        weights: Path to model weights
        data_yaml: Path to dataset YAML
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Loading model: {weights}")
    model = YOLO(str(weights))
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating on dataset: {data_yaml}")
    
    # Run validation
    logger.info("Running validation...")
    results = model.val(
        data=str(data_yaml),
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=True,
        plots=True,
        save_json=True
    )
    
    # Extract metrics
    metrics = {}
    
    try:
        # Detection metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            
            # Get class-wise and overall metrics
            metrics['precision'] = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
            metrics['recall'] = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0
            metrics['mAP50'] = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
            metrics['mAP50-95'] = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
            
            # Compute F1 score
            if metrics['precision'] > 0 or metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0.0
            
            # Per-class metrics if available
            if hasattr(box_metrics, 'ap50') and box_metrics.ap50 is not None:
                metrics['AP50'] = float(box_metrics.ap50[0]) if len(box_metrics.ap50) > 0 else 0.0
            
            if hasattr(box_metrics, 'ap') and box_metrics.ap is not None:
                metrics['AP50-95'] = float(box_metrics.ap[0]) if len(box_metrics.ap) > 0 else 0.0
        
        # Segmentation metrics if available
        if hasattr(results, 'seg'):
            seg_metrics = results.seg
            metrics['seg_precision'] = float(seg_metrics.mp) if hasattr(seg_metrics, 'mp') else 0.0
            metrics['seg_recall'] = float(seg_metrics.mr) if hasattr(seg_metrics, 'mr') else 0.0
            metrics['seg_mAP50'] = float(seg_metrics.map50) if hasattr(seg_metrics, 'map50') else 0.0
            metrics['seg_mAP50-95'] = float(seg_metrics.map) if hasattr(seg_metrics, 'map') else 0.0
            
            if metrics['seg_precision'] > 0 or metrics['seg_recall'] > 0:
                metrics['seg_f1_score'] = 2 * (metrics['seg_precision'] * metrics['seg_recall']) / (metrics['seg_precision'] + metrics['seg_recall'])
            else:
                metrics['seg_f1_score'] = 0.0
        
        # Speed metrics
        if hasattr(results, 'speed'):
            speed = results.speed
            metrics['inference_time_ms'] = speed.get('inference', 0)
            metrics['preprocess_time_ms'] = speed.get('preprocess', 0)
            metrics['postprocess_time_ms'] = speed.get('postprocess', 0)
            metrics['total_time_ms'] = sum([
                metrics['inference_time_ms'],
                metrics['preprocess_time_ms'],
                metrics['postprocess_time_ms']
            ])
            metrics['fps'] = 1000.0 / metrics['total_time_ms'] if metrics['total_time_ms'] > 0 else 0.0
        
        # Model info
        metrics['model_path'] = str(weights)
        metrics['dataset'] = str(data_yaml)
        metrics['conf_threshold'] = conf_threshold
        metrics['iou_threshold'] = iou_threshold
        metrics['device'] = device
        
        # Add quality assessment
        if metrics.get('f1_score', 0) >= 0.8:
            metrics['quality_assessment'] = 'EXCELLENT'
        elif metrics.get('f1_score', 0) >= 0.6:
            metrics['quality_assessment'] = 'GOOD'
        elif metrics.get('f1_score', 0) >= 0.4:
            metrics['quality_assessment'] = 'MODERATE'
        elif metrics.get('f1_score', 0) >= 0.2:
            metrics['quality_assessment'] = 'POOR'
        else:
            metrics['quality_assessment'] = 'VERY_POOR'
        
    except Exception as e:
        logger.error(f"Error extracting metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate pothole detection model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold (default: 0.45)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--output', type=str, default='outputs/metrics.json',
                        help='Output JSON file for metrics (default: outputs/metrics.json)')
    
    args = parser.parse_args()
    
    weights = Path(args.weights)
    data_yaml = Path(args.data)
    
    if not weights.exists():
        logger.error(f"Weights file not found: {weights}")
        return 1
    
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        return 1
    
    # Evaluate model
    metrics = evaluate_model(
        weights=weights,
        data_yaml=data_yaml,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Print metrics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION METRICS")
    logger.info("="*60)
    
    # Detection metrics
    if 'precision' in metrics:
        logger.info("\nDetection Metrics:")
        logger.info(f"  Precision:       {metrics['precision']:.4f}")
        logger.info(f"  Recall:          {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:        {metrics['f1_score']:.4f}")
        logger.info(f"  mAP@0.5:         {metrics['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95:    {metrics['mAP50-95']:.4f}")
        if 'AP50' in metrics:
            logger.info(f"  AP@0.5:          {metrics['AP50']:.4f}")
        if 'AP50-95' in metrics:
            logger.info(f"  AP@0.5:0.95:     {metrics['AP50-95']:.4f}")
    
    # Segmentation metrics
    if 'seg_precision' in metrics:
        logger.info("\nSegmentation Metrics:")
        logger.info(f"  Seg Precision:   {metrics['seg_precision']:.4f}")
        logger.info(f"  Seg Recall:      {metrics['seg_recall']:.4f}")
        logger.info(f"  Seg F1 Score:    {metrics['seg_f1_score']:.4f}")
        logger.info(f"  Seg mAP@0.5:     {metrics['seg_mAP50']:.4f}")
        logger.info(f"  Seg mAP@0.5:0.95:{metrics['seg_mAP50-95']:.4f}")
    
    # Speed metrics
    if 'fps' in metrics:
        logger.info("\nSpeed Metrics:")
        logger.info(f"  Inference:       {metrics['inference_time_ms']:.2f} ms")
        logger.info(f"  Preprocess:      {metrics['preprocess_time_ms']:.2f} ms")
        logger.info(f"  Postprocess:     {metrics['postprocess_time_ms']:.2f} ms")
        logger.info(f"  Total:           {metrics['total_time_ms']:.2f} ms")
        logger.info(f"  FPS:             {metrics['fps']:.2f}")
    
    # Quality assessment
    if 'quality_assessment' in metrics:
        logger.info(f"\nQuality Assessment: {metrics['quality_assessment']}")
    
    logger.info("="*60 + "\n")
    
    # Save metrics to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
