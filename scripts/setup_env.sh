#!/usr/bin/env bash
# Setup script for Pothole Detection Project
# Creates virtual environment and installs dependencies

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Pothole Detection Project - Environment Setup"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check Python version
echo -e "${GREEN}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo "Virtual environment created at .venv/"
else
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${GREEN}Installing Python packages from requirements.txt...${NC}"
if [ -f "requirements.txt" ]; then
    if pip install -r requirements.txt; then
        echo -e "${GREEN}Successfully installed all Python packages.${NC}"
    else
        echo -e "${RED}Error: Failed to install some packages.${NC}"
        echo -e "${YELLOW}If opencv-python fails, try editing requirements.txt to use opencv-python-headless${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

# Check for Homebrew and install ffmpeg (optional)
echo -e "${GREEN}Checking for ffmpeg...${NC}"
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is already installed: $(ffmpeg -version | head -n 1)"
elif command -v brew &> /dev/null; then
    echo -e "${YELLOW}ffmpeg not found. Installing via Homebrew...${NC}"
    if brew install ffmpeg; then
        echo -e "${GREEN}ffmpeg installed successfully.${NC}"
    else
        echo -e "${YELLOW}Warning: Failed to install ffmpeg. Video processing may have issues.${NC}"
    fi
else
    echo -e "${YELLOW}Warning: ffmpeg not found and Homebrew not available.${NC}"
    echo -e "${YELLOW}Video processing will use fallback (imageio-ffmpeg).${NC}"
    echo -e "${YELLOW}For better performance, install ffmpeg manually.${NC}"
fi

# Create necessary directories
echo -e "${GREEN}Creating project directories...${NC}"
mkdir -p configs
mkdir -p outputs
mkdir -p runs
mkdir -p backups
mkdir -p datasets/processed_temp
mkdir -p src
mkdir -p scripts
mkdir -p tests
mkdir -p notebooks

echo ""
echo -e "${GREEN}=========================================="
echo "Setup complete!"
echo "==========================================${NC}"
echo ""
echo "To activate the environment in future sessions, run:"
echo -e "${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo "Next steps:"
echo "  1. Place your videos in datasets/Video_dataset/{train,val,test}/rgb/ and mask/"
echo "  2. Run data preprocessing: python src/data_pipeline.py"
echo "  3. Train model: python src/train_yolo.py --epochs 5"
echo "  4. Run demo: python src/run_demo.py --video datasets/samples/my_road_video.mp4"
echo ""
