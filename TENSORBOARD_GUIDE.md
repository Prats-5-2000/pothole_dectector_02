# TensorBoard Monitoring Guide

This guide explains how to use TensorBoard to monitor YOLOv8 training progress in real-time.

## Quick Start

### 1. Start Training
Training automatically logs metrics to TensorBoard:
```bash
./run_training.sh
```

### 2. Launch TensorBoard
In a separate terminal, run:
```bash
tensorboard --logdir runs/tensorboard --bind_all
```

Then open your browser to: **http://localhost:6006**

## What's Logged

### Loss Metrics (per epoch)
- **Loss/box_loss**: Bounding box regression loss
- **Loss/cls_loss**: Classification loss
- **Loss/dfl_loss**: Distribution focal loss

### Validation Metrics (per epoch)
- **Metrics/mAP@0.5**: Mean Average Precision at IoU=0.5
- **Metrics/mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds
- **Metrics/precision**: Detection precision
- **Metrics/recall**: Detection recall

## Advanced Usage

### Compare Multiple Runs
```bash
# Run 1
./run_training.sh

# Run 2 (change hyperparameters)
python src/train_yolo.py --name yolov8m_det_v2 --batch 16

# View both in TensorBoard
tensorboard --logdir runs/tensorboard --bind_all
```

### Remote Access
If training on a remote server:
```bash
# On remote server
tensorboard --logdir runs/tensorboard --bind_all --port 6006

# On local machine, create SSH tunnel
ssh -L 6006:localhost:6006 user@remote-server
```

Then access at http://localhost:6006

### TensorBoard on Different Port
```bash
tensorboard --logdir runs/tensorboard --bind_all --port 8080
```

## Checkpoints

Training saves checkpoints to `runs/train/yolov8m_det/weights/`:
- **best.pt**: Best model based on validation mAP
- **last.pt**: Most recent epoch
- **epoch_05.pt, epoch_10.pt, ...**: Saved every 5 epochs

## Tips

1. **Real-time monitoring**: TensorBoard updates automatically as training progresses
2. **Smooth curves**: Adjust the "Smoothing" slider in TensorBoard UI
3. **Compare runs**: Select multiple runs from the left sidebar
4. **Download data**: Click "Show data download links" to export CSVs

## Troubleshooting

### TensorBoard not showing data
- Ensure training has completed at least 1 epoch
- Check that `runs/tensorboard/yolov8m_det/` directory exists
- Refresh TensorBoard browser page

### Port already in use
```bash
tensorboard --logdir runs/tensorboard --bind_all --port 6007
```

### View specific experiment
```bash
tensorboard --logdir runs/tensorboard/yolov8m_det --bind_all
```
