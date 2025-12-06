#!/usr/bin/env bash
# =============================================================================
# Monitor Deep Training Pipeline Progress
# =============================================================================

PROJECT_ROOT="/Users/soumyajaiswal/Desktop/Prathamesh Patil/pothole_dectector_02"
LOG_FILE="$PROJECT_ROOT/logs/deep_training_run.log"
CONSOLE_LOG="$PROJECT_ROOT/logs/deep_training_console.log"

echo "================================================"
echo "  DEEP TRAINING PIPELINE MONITOR"
echo "================================================"
echo ""

# Check if process is running
PID=$(pgrep -f "run_deep_retrain.sh")
if [ -n "$PID" ]; then
    echo "✓ Training pipeline is RUNNING (PID: $PID)"
else
    echo "✗ Training pipeline is NOT running"
fi

echo ""
echo "================================================"
echo "  CURRENT PROGRESS"
echo "================================================"
echo ""

# Show last 30 lines of main log
if [ -f "$LOG_FILE" ]; then
    echo "Last updates from training log:"
    echo "---"
    tail -n 30 "$LOG_FILE" | grep -E "\[.*\]|===|Model|Training|Epoch|mAP|Precision|Recall|F1"
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "================================================"
echo "  MONITORING COMMANDS"
echo "================================================"
echo ""
echo "Watch live progress:"
echo "  tail -f \"$LOG_FILE\""
echo ""
echo "Check console output:"
echo "  tail -f \"$CONSOLE_LOG\""
echo ""
echo "Check training metrics (once training starts):"
echo "  watch -n 5 'cat \"$PROJECT_ROOT/runs/detect/model_a_yolov8m_seg/results.csv\" | tail -n 5'"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
echo "================================================"

exit 0
