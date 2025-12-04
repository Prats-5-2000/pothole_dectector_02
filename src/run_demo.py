#!/usr/bin/env python3
"""
Demo Inference Script for Pothole Detection

Runs inference on a video and generates:
- Annotated output video
- JSON report with frame-by-frame detections
- CSV summary
- Simple IoU-based tracking for unique object counting

Author: Pothole Detection Team
Date: 2025-12-04
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SimpleTracker:
    """
    Simple IoU-based tracker for counting unique objects across frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU to associate detection with track
            max_age: Maximum frames to keep track without detection
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[np.ndarray]) -> List[int]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of boxes [x1, y1, x2, y2]
            
        Returns:
            List of track IDs for each detection
        """
        # Increment age of all tracks
        for track in self.tracks:
            track['age'] += 1
        
        if len(detections) == 0:
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
            return []
        
        track_ids = []
        
        for det in detections:
            best_iou = 0
            best_track_idx = -1
            
            # Find best matching track
            for idx, track in enumerate(self.tracks):
                if track['age'] > self.max_age:
                    continue
                
                iou = self.compute_iou(det, track['box'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_idx = idx
            
            if best_track_idx >= 0:
                # Update existing track
                self.tracks[best_track_idx]['box'] = det
                self.tracks[best_track_idx]['age'] = 0
                track_ids.append(self.tracks[best_track_idx]['id'])
            else:
                # Create new track
                new_track = {
                    'id': self.next_id,
                    'box': det,
                    'age': 0
                }
                self.tracks.append(new_track)
                track_ids.append(self.next_id)
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        return track_ids
    
    def get_unique_count(self) -> int:
        """Get total number of unique objects tracked."""
        return self.next_id


def run_inference_on_video(
    model: YOLO,
    video_path: Path,
    output_video_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    tracker_iou: float = 0.3,
    save_first_frame: bool = True
) -> Tuple[List[Dict], pd.DataFrame, int]:
    """
    Run inference on video and generate outputs.
    
    Args:
        model: Loaded YOLO model
        video_path: Input video path
        output_video_path: Output video path
        conf_threshold: Confidence threshold for detections
        iou_threshold: NMS IoU threshold
        tracker_iou: IoU threshold for tracking
        save_first_frame: Save first frame as image
        
    Returns:
        Tuple of (frame_detections, summary_df, unique_count)
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Setup output video writer
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = SimpleTracker(iou_threshold=tracker_iou)
    
    # Storage for results
    frame_detections = []
    summary_data = []
    
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save first frame
            if frame_idx == 0 and save_first_frame:
                first_frame_path = output_video_path.parent / f"{output_video_path.stem}_first_frame.jpg"
                cv2.imwrite(str(first_frame_path), frame)
                logger.info(f"First frame saved to: {first_frame_path}")
            
            # Run inference
            results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # Extract detections
            detections = []
            boxes_xyxy = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'box': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls
                        })
                        boxes_xyxy.append(np.array([x1, y1, x2, y2]))
            
            # Update tracker
            track_ids = tracker.update(boxes_xyxy) if boxes_xyxy else []
            
            # Add track IDs to detections
            for det, track_id in zip(detections, track_ids):
                det['track_id'] = int(track_id)
            
            # Store frame data
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'num_detections': len(detections),
                'detections': detections
            }
            frame_detections.append(frame_data)
            
            summary_data.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'num_detections': len(detections)
            })
            
            # Draw on frame
            annotated_frame = frame.copy()
            
            for det in detections:
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                track_id = det.get('track_id', -1)
                
                # Draw box
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"ID:{track_id} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1) - label_size[1] - 4),
                    (int(x1) + label_size[0], int(y1)),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
            
            # Add info text
            info_text = f"Frame: {frame_idx} | Detections: {len(detections)} | Unique: {tracker.get_unique_count()}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Write frame
            out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    logger.info(f"Output video saved to: {output_video_path}")
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    unique_count = tracker.get_unique_count()
    logger.info(f"Total unique objects tracked: {unique_count}")
    
    return frame_detections, summary_df, unique_count


def save_reports(
    frame_detections: List[Dict],
    summary_df: pd.DataFrame,
    unique_count: int,
    output_dir: Path,
    video_name: str
):
    """
    Save JSON and CSV reports.
    
    Args:
        frame_detections: Frame-by-frame detection data
        summary_df: Summary dataframe
        unique_count: Total unique objects tracked
        output_dir: Output directory
        video_name: Base name for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / f"{video_name}_report.json"
    report = {
        'summary': {
            'total_frames': len(frame_detections),
            'total_detections': sum(f['num_detections'] for f in frame_detections),
            'unique_objects': unique_count,
            'avg_detections_per_frame': np.mean([f['num_detections'] for f in frame_detections])
        },
        'frames': frame_detections
    }
    
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"JSON report saved to: {json_path}")
    
    # Save CSV summary
    csv_path = output_dir / f"{video_name}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"CSV summary saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run pothole detection inference on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with trained weights
  python src/run_demo.py --weights runs/detect/video_merged/weights/best.pt --video datasets/samples/my_road_video.mp4

  # Adjust detection thresholds
  python src/run_demo.py --weights best.pt --video my_video.mp4 --conf 0.3 --iou 0.4

  # Custom output location
  python src/run_demo.py --weights best.pt --video my_video.mp4 --output custom_outputs/
        """
    )
    
    parser.add_argument(
        '--weights',
        type=Path,
        required=True,
        help='Path to trained YOLO weights (.pt file)'
    )
    
    parser.add_argument(
        '--video',
        type=Path,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs'),
        help='Output directory (default: outputs/)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='NMS IoU threshold (default: 0.45)'
    )
    
    parser.add_argument(
        '--tracker-iou',
        type=float,
        default=0.3,
        help='Tracker IoU threshold (default: 0.3)'
    )
    
    parser.add_argument(
        '--no-first-frame',
        action='store_true',
        help='Do not save first frame image'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check inputs
    if not args.weights.exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)
    
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from: {args.weights}")
    try:
        model = YOLO(str(args.weights))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    logger.info("Model loaded successfully")
    
    # Setup output paths
    video_basename = args.video.stem
    output_video = args.output / f"{video_basename}_out.mp4"
    
    # Run inference
    logger.info(f"Running inference on: {args.video}")
    frame_detections, summary_df, unique_count = run_inference_on_video(
        model=model,
        video_path=args.video,
        output_video_path=output_video,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_iou=args.tracker_iou,
        save_first_frame=not args.no_first_frame
    )
    
    # Save reports
    save_reports(
        frame_detections=frame_detections,
        summary_df=summary_df,
        unique_count=unique_count,
        output_dir=args.output,
        video_name=video_basename
    )
    
    logger.info("\n" + "="*50)
    logger.info("Demo inference completed!")
    logger.info("="*50)
    logger.info(f"\nOutputs saved to: {args.output}/")
    logger.info(f"  - Annotated video: {output_video.name}")
    logger.info(f"  - JSON report: {video_basename}_report.json")
    logger.info(f"  - CSV summary: {video_basename}_summary.csv")
    if not args.no_first_frame:
        logger.info(f"  - First frame: {video_basename}_first_frame.jpg")


if __name__ == '__main__':
    main()
