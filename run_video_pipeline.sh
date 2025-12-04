#!/usr/bin/env bash
# =============================================================================
# Pothole Detection Video Pipeline - Single Command Execution
# =============================================================================
#
# This script runs the complete pothole detection pipeline on a video:
# 1. Activates virtual environment
# 2. Finds newest trained weights
# 3. Validates and re-encodes video if needed
# 4. Runs inference
# 5. If zero detections, runs auto fine-tuning and re-runs inference
# 6. Outputs summary report
#
# Usage: ./run_video_pipeline.sh
#
# Author: Pothole Detection Team
# Date: 2025-12-04
# =============================================================================

set -e  # Exit on error (except where we handle it explicitly)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VIDEO_PATH="datasets/samples/my_road_video.mp4"
OUTPUT_DIR="outputs"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/video_pipeline.log"
BACKUP_DIR="backups"

DEMO_VIDEO_OUTPUT="$OUTPUT_DIR/demo_video_final.mp4"
DEMO_REPORT_JSON="$OUTPUT_DIR/demo_report.json"
DEMO_FIRST_FRAME="$OUTPUT_DIR/demo_first_frame.jpg"
DEMO_SUMMARY_CSV="$OUTPUT_DIR/demo_summary.csv"

FINETUNE_DIR="runs/detect/video_finetuned"
FINETUNE_WEIGHTS="$FINETUNE_DIR/weights/final_best.pt"

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
}

# =============================================================================
# Setup and Initialization
# =============================================================================

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BACKUP_DIR"

# Clear previous log
> "$LOG_FILE"

log_header "POTHOLE DETECTION VIDEO PIPELINE"
log "Pipeline started at: $(date)"
log "Project root: $PROJECT_ROOT"
log "Video path: $VIDEO_PATH"
log ""

# =============================================================================
# Step 1: Activate Virtual Environment
# =============================================================================

log_header "Step 1: Activating Virtual Environment"

if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found at .venv/"
    log_error "Please run: ./scripts/setup_env.sh first"
    exit 1
fi

source .venv/bin/activate
log_success "Virtual environment activated"

# Verify Python
PYTHON_VERSION=$(python --version 2>&1)
log "Python version: $PYTHON_VERSION"

# =============================================================================
# Step 2: Find Newest Weights
# =============================================================================

log_header "Step 2: Finding Newest Trained Weights"

if [ ! -d "runs/detect" ]; then
    log_error "No training runs found at runs/detect/"
    log_error "Please train a model first: python src/train_yolo.py --epochs 5"
    exit 1
fi

WEIGHTS_PATH=$(python src/find_weights.py)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ -z "$WEIGHTS_PATH" ]; then
    log_error "Failed to find weights"
    log_error "Please train a model first: python src/train_yolo.py --epochs 5"
    exit 1
fi

log_success "Found weights: $WEIGHTS_PATH"

# Backup weights
WEIGHTS_BACKUP="$BACKUP_DIR/$(basename $WEIGHTS_PATH .pt)_$(date +%Y%m%d_%H%M%S).pt"
cp "$WEIGHTS_PATH" "$WEIGHTS_BACKUP"
log "Backup created: $WEIGHTS_BACKUP"

# =============================================================================
# Step 3: Video Sanity Check
# =============================================================================

log_header "Step 3: Video Sanity Check"

if [ ! -f "$VIDEO_PATH" ]; then
    log_error "Video not found: $VIDEO_PATH"
    exit 1
fi

log "Checking video: $VIDEO_PATH"

# Check video with OpenCV and re-encode if needed
PROCESSED_VIDEO=$(python src/check_video.py --video "$VIDEO_PATH")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log_error "Video check failed"
    exit 1
fi

if [ "$PROCESSED_VIDEO" != "$VIDEO_PATH" ]; then
    log_warning "Video was re-encoded to: $PROCESSED_VIDEO"
    VIDEO_TO_USE="$PROCESSED_VIDEO"
else
    log_success "Video is compatible with OpenCV"
    VIDEO_TO_USE="$VIDEO_PATH"
fi

# =============================================================================
# Step 4: Run Initial Inference
# =============================================================================

log_header "Step 4: Running Initial Inference"

log "Running inference on: $(basename $VIDEO_TO_USE)"
log "Using weights: $WEIGHTS_PATH"

# Temporary output paths for first inference
TEMP_VIDEO_OUT="$OUTPUT_DIR/temp_video_out.mp4"
TEMP_REPORT_JSON="$OUTPUT_DIR/temp_report.json"
TEMP_SUMMARY_CSV="$OUTPUT_DIR/temp_summary.csv"
TEMP_FIRST_FRAME="$OUTPUT_DIR/temp_first_frame.jpg"

python src/run_demo.py \
    --weights "$WEIGHTS_PATH" \
    --video "$VIDEO_TO_USE" \
    --output "$OUTPUT_DIR" \
    --conf 0.25 \
    --iou 0.45 \
    --tracker-iou 0.3 2>&1 | tee -a "$LOG_FILE"

INFERENCE_EXIT=$?

if [ $INFERENCE_EXIT -ne 0 ]; then
    log_error "Inference failed"
    exit 1
fi

log_success "Initial inference completed"

# Rename outputs to temp names
if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_out.mp4" ]; then
    mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_out.mp4" "$TEMP_VIDEO_OUT"
fi
if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_report.json" ]; then
    mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_report.json" "$TEMP_REPORT_JSON"
fi
if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_summary.csv" ]; then
    mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_summary.csv" "$TEMP_SUMMARY_CSV"
fi
if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_first_frame.jpg" ]; then
    mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_first_frame.jpg" "$TEMP_FIRST_FRAME"
fi

# =============================================================================
# Step 5: Check Detection Count and Auto Fine-tune if Needed
# =============================================================================

log_header "Step 5: Analyzing Detection Results"

# Parse JSON report to get detection count
if [ -f "$TEMP_REPORT_JSON" ]; then
    TOTAL_DETECTIONS=$(python3 -c "
import json
import sys
try:
    with open('$TEMP_REPORT_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('summary', {}).get('total_detections', 0))
except:
    print(0)
")
    
    UNIQUE_OBJECTS=$(python3 -c "
import json
import sys
try:
    with open('$TEMP_REPORT_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('summary', {}).get('unique_objects', 0))
except:
    print(0)
")
else
    TOTAL_DETECTIONS=0
    UNIQUE_OBJECTS=0
fi

log "Total detections: $TOTAL_DETECTIONS"
log "Unique objects tracked: $UNIQUE_OBJECTS"

if [ "$TOTAL_DETECTIONS" -eq 0 ]; then
    log_warning "Zero detections found - initiating auto fine-tuning"
    
    log_header "Step 5a: Auto Fine-tuning"
    
    # Create a small dataset for fine-tuning from video frames
    log "Extracting frames from video for fine-tuning..."
    
    FINETUNE_UNLABELED="datasets/processed_temp/finetune_frames"
    mkdir -p "$FINETUNE_UNLABELED"
    
    # Extract frames using ffmpeg (every 30th frame)
    ffmpeg -i "$VIDEO_TO_USE" -vf "select='not(mod(n\,30))'" -vsync vfr \
        "$FINETUNE_UNLABELED/frame_%04d.jpg" -loglevel error 2>&1 | tee -a "$LOG_FILE"
    
    FRAME_COUNT=$(ls -1 "$FINETUNE_UNLABELED"/*.jpg 2>/dev/null | wc -l || echo "0")
    log "Extracted $FRAME_COUNT frames for fine-tuning"
    
    if [ "$FRAME_COUNT" -gt 0 ]; then
        # Run auto fine-tuning
        log "Running auto fine-tuning (this may take several minutes)..."
        
        python src/auto_finetune.py \
            --weights "$WEIGHTS_PATH" \
            --unlabeled "$FINETUNE_UNLABELED" \
            --output "datasets/finetune" \
            --backup "$BACKUP_DIR" \
            --iterations 1 \
            --pseudo-conf 0.4 \
            --min-box-area 150 \
            --epochs 3 \
            --batch 4 \
            --device cpu 2>&1 | tee -a "$LOG_FILE"
        
        FINETUNE_EXIT=$?
        
        if [ $FINETUNE_EXIT -eq 0 ]; then
            # Find fine-tuned weights
            FINETUNED_WEIGHTS=$(find datasets/finetune -name "best.pt" -type f | head -n 1)
            
            if [ -n "$FINETUNED_WEIGHTS" ] && [ -f "$FINETUNED_WEIGHTS" ]; then
                log_success "Fine-tuning completed: $FINETUNED_WEIGHTS"
                
                # Create final weights directory
                mkdir -p "$(dirname $FINETUNE_WEIGHTS)"
                cp "$FINETUNED_WEIGHTS" "$FINETUNE_WEIGHTS"
                log_success "Fine-tuned weights saved to: $FINETUNE_WEIGHTS"
                
                # Re-run inference with fine-tuned weights
                log_header "Step 5b: Re-running Inference with Fine-tuned Weights"
                
                python src/run_demo.py \
                    --weights "$FINETUNE_WEIGHTS" \
                    --video "$VIDEO_TO_USE" \
                    --output "$OUTPUT_DIR" \
                    --conf 0.25 \
                    --iou 0.45 \
                    --tracker-iou 0.3 2>&1 | tee -a "$LOG_FILE"
                
                RERUN_EXIT=$?
                
                if [ $RERUN_EXIT -eq 0 ]; then
                    log_success "Re-inference completed with fine-tuned weights"
                    WEIGHTS_USED="$FINETUNE_WEIGHTS (fine-tuned)"
                    
                    # Move new outputs
                    if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_out.mp4" ]; then
                        mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_out.mp4" "$DEMO_VIDEO_OUTPUT"
                    fi
                    if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_report.json" ]; then
                        mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_report.json" "$DEMO_REPORT_JSON"
                    fi
                    if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_summary.csv" ]; then
                        mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_summary.csv" "$DEMO_SUMMARY_CSV"
                    fi
                    if [ -f "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_first_frame.jpg" ]; then
                        mv "$OUTPUT_DIR/$(basename $VIDEO_TO_USE .mp4)_first_frame.jpg" "$DEMO_FIRST_FRAME"
                    fi
                    
                    # Update detection counts
                    if [ -f "$DEMO_REPORT_JSON" ]; then
                        TOTAL_DETECTIONS=$(python3 -c "
import json
try:
    with open('$DEMO_REPORT_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('summary', {}).get('total_detections', 0))
except:
    print(0)
")
                        UNIQUE_OBJECTS=$(python3 -c "
import json
try:
    with open('$DEMO_REPORT_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('summary', {}).get('unique_objects', 0))
except:
    print(0)
")
                    fi
                else
                    log_error "Re-inference failed, keeping original results"
                    WEIGHTS_USED="$WEIGHTS_PATH (original)"
                    # Use temp outputs as final
                    [ -f "$TEMP_VIDEO_OUT" ] && mv "$TEMP_VIDEO_OUT" "$DEMO_VIDEO_OUTPUT"
                    [ -f "$TEMP_REPORT_JSON" ] && mv "$TEMP_REPORT_JSON" "$DEMO_REPORT_JSON"
                    [ -f "$TEMP_SUMMARY_CSV" ] && mv "$TEMP_SUMMARY_CSV" "$DEMO_SUMMARY_CSV"
                    [ -f "$TEMP_FIRST_FRAME" ] && mv "$TEMP_FIRST_FRAME" "$DEMO_FIRST_FRAME"
                fi
            else
                log_warning "Fine-tuned weights not found, using original results"
                WEIGHTS_USED="$WEIGHTS_PATH (original)"
                # Use temp outputs as final
                [ -f "$TEMP_VIDEO_OUT" ] && mv "$TEMP_VIDEO_OUT" "$DEMO_VIDEO_OUTPUT"
                [ -f "$TEMP_REPORT_JSON" ] && mv "$TEMP_REPORT_JSON" "$DEMO_REPORT_JSON"
                [ -f "$TEMP_SUMMARY_CSV" ] && mv "$TEMP_SUMMARY_CSV" "$DEMO_SUMMARY_CSV"
                [ -f "$TEMP_FIRST_FRAME" ] && mv "$TEMP_FIRST_FRAME" "$DEMO_FIRST_FRAME"
            fi
        else
            log_warning "Fine-tuning failed, using original results"
            WEIGHTS_USED="$WEIGHTS_PATH (original)"
            # Use temp outputs as final
            [ -f "$TEMP_VIDEO_OUT" ] && mv "$TEMP_VIDEO_OUT" "$DEMO_VIDEO_OUTPUT"
            [ -f "$TEMP_REPORT_JSON" ] && mv "$TEMP_REPORT_JSON" "$DEMO_REPORT_JSON"
            [ -f "$TEMP_SUMMARY_CSV" ] && mv "$TEMP_SUMMARY_CSV" "$DEMO_SUMMARY_CSV"
            [ -f "$TEMP_FIRST_FRAME" ] && mv "$TEMP_FIRST_FRAME" "$DEMO_FIRST_FRAME"
        fi
    else
        log_warning "No frames extracted, using original results"
        WEIGHTS_USED="$WEIGHTS_PATH (original)"
        # Use temp outputs as final
        [ -f "$TEMP_VIDEO_OUT" ] && mv "$TEMP_VIDEO_OUT" "$DEMO_VIDEO_OUTPUT"
        [ -f "$TEMP_REPORT_JSON" ] && mv "$TEMP_REPORT_JSON" "$DEMO_REPORT_JSON"
        [ -f "$TEMP_SUMMARY_CSV" ] && mv "$TEMP_SUMMARY_CSV" "$DEMO_SUMMARY_CSV"
        [ -f "$TEMP_FIRST_FRAME" ] && mv "$TEMP_FIRST_FRAME" "$DEMO_FIRST_FRAME"
    fi
else
    log_success "Detections found, no fine-tuning needed"
    WEIGHTS_USED="$WEIGHTS_PATH (original)"
    # Use temp outputs as final
    [ -f "$TEMP_VIDEO_OUT" ] && mv "$TEMP_VIDEO_OUT" "$DEMO_VIDEO_OUTPUT"
    [ -f "$TEMP_REPORT_JSON" ] && mv "$TEMP_REPORT_JSON" "$DEMO_REPORT_JSON"
    [ -f "$TEMP_SUMMARY_CSV" ] && mv "$TEMP_SUMMARY_CSV" "$DEMO_SUMMARY_CSV"
    [ -f "$TEMP_FIRST_FRAME" ] && mv "$TEMP_FIRST_FRAME" "$DEMO_FIRST_FRAME"
fi

# =============================================================================
# Step 6: Final Summary
# =============================================================================

log_header "PIPELINE SUMMARY"

# Get frame count from JSON
TOTAL_FRAMES=0
if [ -f "$DEMO_REPORT_JSON" ]; then
    TOTAL_FRAMES=$(python3 -c "
import json
try:
    with open('$DEMO_REPORT_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('summary', {}).get('total_frames', 0))
except:
    print(0)
")
fi

echo "" | tee -a "$LOG_FILE"
echo -e "${GREEN}Pipeline completed successfully!${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${CYAN}Weights Used:${NC} $WEIGHTS_USED" | tee -a "$LOG_FILE"
echo -e "${CYAN}Total Frames Processed:${NC} $TOTAL_FRAMES" | tee -a "$LOG_FILE"
echo -e "${CYAN}Total Potholes Detected:${NC} $TOTAL_DETECTIONS" | tee -a "$LOG_FILE"
echo -e "${CYAN}Unique Potholes Tracked:${NC} $UNIQUE_OBJECTS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${CYAN}Output Files:${NC}" | tee -a "$LOG_FILE"

if [ -f "$DEMO_VIDEO_OUTPUT" ]; then
    echo -e "  ${GREEN}✓${NC} Annotated video: $DEMO_VIDEO_OUTPUT" | tee -a "$LOG_FILE"
else
    echo -e "  ${RED}✗${NC} Annotated video: NOT FOUND" | tee -a "$LOG_FILE"
fi

if [ -f "$DEMO_REPORT_JSON" ]; then
    echo -e "  ${GREEN}✓${NC} JSON report: $DEMO_REPORT_JSON" | tee -a "$LOG_FILE"
else
    echo -e "  ${RED}✗${NC} JSON report: NOT FOUND" | tee -a "$LOG_FILE"
fi

if [ -f "$DEMO_SUMMARY_CSV" ]; then
    echo -e "  ${GREEN}✓${NC} CSV summary: $DEMO_SUMMARY_CSV" | tee -a "$LOG_FILE"
else
    echo -e "  ${RED}✗${NC} CSV summary: NOT FOUND" | tee -a "$LOG_FILE"
fi

if [ -f "$DEMO_FIRST_FRAME" ]; then
    echo -e "  ${GREEN}✓${NC} First frame: $DEMO_FIRST_FRAME" | tee -a "$LOG_FILE"
else
    echo -e "  ${RED}✗${NC} First frame: NOT FOUND" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo -e "${CYAN}Complete log saved to:${NC} $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log "Pipeline finished at: $(date)"

exit 0
