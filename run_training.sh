#!/bin/bash
################################################################################
# run_training.sh - Clean YOLOv8m-seg Training Script
# 
# Purpose: Train YOLOv8m-seg model on merged pothole dataset with live monitoring
# 
# Features:
#   - Clean command without unsupported arguments
#   - Live training progress updates per epoch
#   - Logs saved to logs/training_yolov8m_seg.log
#   - Both terminal output and log file capture
# 
# Usage:
#   ./run_training.sh
################################################################################

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}   YOLOv8m-seg Training Script   ${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Found virtual environment (.venv)"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Found virtual environment (venv)"
    source venv/bin/activate
elif [ -d "env" ]; then
    echo -e "${GREEN}✓${NC} Found virtual environment (env)"
    source env/bin/activate
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found"
    echo "  Proceeding without venv activation..."
fi

# Verify Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗${NC} python3 not found. Please install Python 3.8+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python: $(python3 --version)"

# Check if data config exists
if [ ! -f "configs/dataset_video.yaml" ]; then
    echo -e "${RED}✗${NC} Dataset config not found: configs/dataset_video.yaml"
    echo "  Please run dataset preparation first"
    exit 1
fi

echo -e "${GREEN}✓${NC} Dataset config found"

# Create logs directory
mkdir -p logs

# Clear previous log
LOG_FILE="logs/training_yolov8m_det.log"
> "$LOG_FILE"
echo -e "${GREEN}✓${NC} Log file: $LOG_FILE"

echo ""
echo -e "${BLUE}Starting training...${NC}"
echo ""
# Detect device (prefer MPS on macOS)
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
fi

echo "Configuration:"
echo "  Model:      yolov8m.pt (detection)"
echo "  Dataset:    configs/dataset_video.yaml"
echo "  Epochs:     30"
echo "  Image size: 640"
echo "  Batch size: 8"
echo "  Device:     $DEVICE"
echo "  Output:     runs/train/yolov8m_det"
echo ""

# Training command (clean, without unsupported args)
# This will use the callback system in train_yolo.py for live monitoring
python3 src/train_yolo.py \
    --model yolov8m.pt \
    --data configs/dataset_video.yaml \
    --epochs 30 \
    --imgsz 640 \
    --batch 8 \
    --device "$DEVICE" \
    --project runs/train \
    --name yolov8m_det \
    --patience 5 \
    --workers 4 \
    --verbose \
    2>&1 | tee -a "$LOG_FILE"

# Check training status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}  ✓ Training Completed!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo "Outputs saved to:"
    echo "  Weights:  runs/train/yolov8m_seg/weights/"
    echo "  Plots:    runs/train/yolov8m_seg/"
    echo "  Log file: $LOG_FILE"
    echo ""
    
    # Show best weights if they exist
    if [ -f "runs/train/yolov8m_seg/weights/best.pt" ]; then
        BEST_SIZE=$(du -h "runs/train/yolov8m_seg/weights/best.pt" | cut -f1)
        echo -e "${GREEN}✓${NC} Best weights: runs/train/yolov8m_seg/weights/best.pt ($BEST_SIZE)"
    fi
    
    if [ -f "runs/train/yolov8m_seg/weights/last.pt" ]; then
        LAST_SIZE=$(du -h "runs/train/yolov8m_seg/weights/last.pt" | cut -f1)
        echo -e "${GREEN}✓${NC} Last weights: runs/train/yolov8m_seg/weights/last.pt ($LAST_SIZE)"
    fi
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}  ✗ Training Failed${NC}"
    echo -e "${RED}=================================${NC}"
    echo ""
    echo "Check log file for details: $LOG_FILE"
    exit 1
fi
