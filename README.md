# Pothole Detection Project

A complete, production-ready pothole detection system using YOLOv8. This project provides tools for dataset preparation, model training, inference, and automatic fine-tuning.

## Features

- **CPU-first design**: Works on Mac/CPU with automatic GPU detection
- **Complete pipeline**: Frame extraction, label conversion, training, and inference
- **Multiple dataset support**: Video datasets, Pascal VOC XML annotations
- **Auto fine-tuning**: Iterative improvement with pseudo-labeling
- **Comprehensive outputs**: Annotated videos, JSON reports, CSV summaries, tracking
- **Production-ready code**: Clean, documented, tested, with CLI interfaces

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── defaults.yaml       # Default hyperparameters
│   └── dataset_video.yaml  # Auto-generated dataset config
├── datasets/               # Dataset directory
│   ├── Video_dataset/      # Main video dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── AndrewMVD pothole dataset/  # Optional Pascal VOC dataset
│   ├── merged/             # Auto-generated merged dataset
│   └── samples/            # Sample videos for testing
├── src/                    # Source code
│   ├── data_pipeline.py    # Data preprocessing
│   ├── train_yolo.py       # Model training
│   ├── run_demo.py         # Inference and demo
│   └── auto_finetune.py    # Auto fine-tuning
├── scripts/                # Setup scripts
│   └── setup_env.sh        # Environment setup
├── tests/                  # Unit tests
│   └── test_data_pipeline.py
├── notebooks/              # Jupyter notebooks
│   └── quick_test.ipynb
├── runs/                   # Training runs (auto-generated)
├── outputs/                # Inference outputs (auto-generated)
├── backups/                # Model backups (auto-generated)
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Environment Setup

Run the setup script to create virtual environment and install dependencies:

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 2. Prepare Dataset

Place your data in the following structure:

```
datasets/
  Video_dataset/
    train/
      rgb/         # Training videos
      mask/        # Mask videos
    val/
      rgb/
      mask/
    test/
      rgb/
      mask/
```

Run data preprocessing (dry-run for testing):

```bash
# Process only first video per split
python src/data_pipeline.py --dry-run

# Process full dataset
python src/data_pipeline.py --fps 1 --min-area 100
```

This will:
- Extract frames from videos at 1 FPS
- Convert mask videos to YOLO detection labels
- Convert Pascal VOC XMLs (if present) to YOLO format
- Create merged dataset at `datasets/merged/`
- Generate `configs/dataset_video.yaml`

### 3. Train Model

Train YOLOv8n for quick testing (5 epochs):

```bash
python src/train_yolo.py --epochs 5 --batch 8
```

Full training (50 epochs):

```bash
python src/train_yolo.py --epochs 50 --batch 16 --imgsz 640
```

Training options:
- `--model`: Model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- `--epochs`: Number of training epochs
- `--batch`: Batch size (auto-adjusted for CPU)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device (auto, cuda, mps, cpu)
- `--tiny`: Quick test mode (yolov8n, 320px, batch=8)
- `--debug`: Debug mode (2 epochs for testing)

Trained weights are saved to: `runs/detect/video_merged/weights/best.pt`

### 4. Run Demo Inference

Run inference on a video:

```bash
python src/run_demo.py \
    --weights runs/detect/video_merged/weights/best.pt \
    --video datasets/samples/my_road_video.mp4 \
    --output outputs/
```

This generates:
- `outputs/my_road_video_out.mp4` - Annotated video with bounding boxes
- `outputs/my_road_video_report.json` - Frame-by-frame detections
- `outputs/my_road_video_summary.csv` - Summary statistics
- `outputs/my_road_video_first_frame.jpg` - First frame with detections

Options:
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.45)
- `--tracker-iou`: Tracking IoU threshold (default: 0.3)

## Advanced Usage

### Auto Fine-tuning with Pseudo-labeling

Improve model performance using unlabeled data:

```bash
python src/auto_finetune.py \
    --weights runs/detect/video_merged/weights/best.pt \
    --unlabeled datasets/samples/ \
    --iterations 2 \
    --pseudo-conf 0.5 \
    --epochs 5
```

This will:
1. Run inference on unlabeled images
2. Filter high-confidence predictions as pseudo-labels
3. Fine-tune model on pseudo-labeled data
4. Evaluate improvement and keep best model
5. Backup previous weights to `backups/`

Safety features:
- Maximum 2 iterations (configurable)
- Conservative confidence thresholds
- Aborts if no improvement or metrics degrade
- Automatic weight backups

### Custom Model Variants

Train with different YOLO models:

```bash
# Larger model for better accuracy
python src/train_yolo.py --model yolov8s.pt --epochs 50

# Tiny model for fast inference
python src/train_yolo.py --tiny --epochs 20
```

### Adjust Detection Parameters

Fine-tune detection thresholds:

```bash
python src/run_demo.py \
    --weights best.pt \
    --video test.mp4 \
    --conf 0.3 \
    --iou 0.4 \
    --tracker-iou 0.25
```

### Clean Up Temporary Files

Remove temporary frame directories after processing:

```bash
python src/data_pipeline.py --clean
```

## Testing

Run unit tests:

```bash
python tests/test_data_pipeline.py
```

Try the interactive notebook:

```bash
jupyter notebook notebooks/quick_test.ipynb
```

## Dataset Format

### Video Dataset Structure

```
Video_dataset/
  {split}/           # train, val, or test
    rgb/             # RGB videos
      video1.mp4
      video2.mp4
    mask/            # Mask videos (same names as rgb)
      video1.mp4
      video2.mp4
```

### Pascal VOC Dataset (AndrewMVD)

```
AndrewMVD pothole dataset/
  images/
    pothole001.jpg
    pothole002.jpg
  annotations/
    pothole001.xml
    pothole002.xml
```

### YOLO Format Output

The pipeline generates YOLO format labels:

```
merged/
  images/
    train/
      frame_000001.jpg
    val/
      frame_000002.jpg
    test/
      frame_000003.jpg
  labels/
    train/
      frame_000001.txt  # class x_center y_center width height (normalized)
    val/
      frame_000002.txt
    test/
      frame_000003.txt
```

## Configuration

Edit `configs/defaults.yaml` to customize default parameters:

```yaml
data_pipeline:
  fps: 1
  min_area: 100

training:
  model: "yolov8n.pt"
  epochs: 50
  batch_size: 16

inference:
  conf_threshold: 0.25
  iou_threshold: 0.45

finetune:
  max_iterations: 2
  pseudo_conf: 0.5
```

## Output Examples

### JSON Report Structure

```json
{
  "summary": {
    "total_frames": 300,
    "total_detections": 42,
    "unique_objects": 15,
    "avg_detections_per_frame": 0.14
  },
  "frames": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "num_detections": 2,
      "detections": [
        {
          "box": [100.5, 200.3, 150.2, 250.8],
          "confidence": 0.87,
          "class": 0,
          "track_id": 1
        }
      ]
    }
  ]
}
```

### CSV Summary

```csv
frame_idx,timestamp,num_detections
0,0.0,2
1,0.033,2
2,0.066,1
```

## Troubleshooting

### Import Errors

If you see import errors, make sure the virtual environment is activated:

```bash
source .venv/bin/activate
```

### OpenCV Issues on Headless Systems

Edit `requirements.txt` to use `opencv-python-headless` instead of `opencv-python`.

### Low Disk Space Warning

The pipeline checks for at least 5GB free space. Clear space or process smaller batches with `--dry-run`.

### Video Processing Issues

Install ffmpeg for better video handling:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### CPU Training Too Slow

Use smaller model and batch size:

```bash
python src/train_yolo.py --tiny --batch 4 --epochs 10
```

## Manual Labeling

For manual annotation, use tools like:
- [LabelImg](https://github.com/heartexlabs/labelImg) - Pascal VOC format
- [Roboflow](https://roboflow.com/) - Online annotation
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool

Export labels in YOLO format or Pascal VOC (which can be converted by this pipeline).

## Performance Tips

1. **Use GPU**: Automatic detection, but ensure PyTorch with CUDA/MPS support is installed
2. **Adjust batch size**: Larger batches = faster training (if memory allows)
3. **Image size**: 640 is default, use 320 for faster inference
4. **FPS extraction**: Lower FPS = fewer frames = faster processing
5. **Dry-run first**: Test with `--dry-run` before full dataset processing

## Assumptions

- Videos in `datasets/Video_dataset/{split}/rgb/` are RGB videos
- Mask videos have same names as RGB videos in `mask/` subdirectory
- Mask videos are binary (white=pothole, black=background)
- Pascal VOC XMLs contain bounding boxes for class "pothole"
- All outputs use class ID 0 for potholes

## Command Reference

### Complete Example Workflow

```bash
# 1. Setup environment
./scripts/setup_env.sh
source .venv/bin/activate

# 2. Dry-run preprocessing (test with 1 video per split)
python src/data_pipeline.py --dry-run

# 3. Quick training (5 epochs)
python src/train_yolo.py --epochs 5 --batch 8

# 4. Run demo
python src/run_demo.py \
    --weights runs/detect/video_merged/weights/best.pt \
    --video datasets/samples/test_video.mp4

# 5. Full processing and training
python src/data_pipeline.py
python src/train_yolo.py --epochs 50

# 6. Auto fine-tune (optional)
python src/auto_finetune.py \
    --weights runs/detect/video_merged/weights/best.pt \
    --unlabeled datasets/samples/
```

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review `configs/defaults.yaml` for configuration options
3. Run with `--help` flag for any script to see all options
4. Check logs in the console output

## Contributors

Pothole Detection Team - 2025