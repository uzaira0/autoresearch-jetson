#!/usr/bin/env bash
# Autoresearch benchmark runner for FLASH-TV inference speed
# Runs the pipeline on a test video, measures throughput
# Outputs METRIC lines for the agent to parse
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Configuration (EDIT THESE before first run) ─────────────────────
TEST_VIDEO="${TEST_VIDEO:-/path/to/test_video.mp4}"       # Test video file
FAMILY_ID="${FAMILY_ID:-999}"                               # Test family ID
DATA_DIR="${DATA_DIR:-/tmp/autoresearch_flash_output}"      # Output directory
USERNAME="${USERNAME:-$(whoami)}"                            # System username
PYTHON="${PYTHON:-python3}"                                 # Python binary
RUNS=3                                                      # Number of benchmark runs

# Validate test video exists
if [ ! -f "$TEST_VIDEO" ]; then
  echo "FATAL: Test video not found: $TEST_VIDEO"
  echo "Set TEST_VIDEO=/path/to/video before running"
  exit 1
fi

# Clean output directory
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# ── Benchmark ───────────────────────────────────────────────────────
echo "=== FLASH-TV Inference Benchmark ($RUNS runs) ==="
BEST_FPS=0
BEST_FRAME_MS=999999

for i in $(seq 1 $RUNS); do
  rm -rf "$DATA_DIR"/*

  # Measure startup + inference time
  START_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

  # Run the pipeline on test video
  # NOTE: Adapt this command to match your actual invocation
  OUTPUT=$($PYTHON python_scripts/test_vid_frames_batch_v7_2fps_frminp_newfv_rotate.py \
    "$FAMILY_ID" "$DATA_DIR" save-image "$USERNAME" \
    --test-video "$TEST_VIDEO" 2>&1) || true

  END_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
  DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))

  # Extract frame count from output or log files
  FRAME_COUNT=$(echo "$OUTPUT" | grep -oE "Processed [0-9]+ frames" | grep -oE "[0-9]+" || echo "0")
  if [ "$FRAME_COUNT" = "0" ]; then
    # Try counting log lines (each line = one processed frame batch)
    LOG_FILE=$(find "$DATA_DIR" -name "*flash_log*" -not -name "*reg*" -not -name "*rot*" | head -1)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
      FRAME_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
    fi
  fi

  # Calculate FPS
  if [ "$FRAME_COUNT" -gt 0 ] && [ "$DURATION_MS" -gt 0 ]; then
    # Use python for float division
    FPS=$($PYTHON -c "print(f'{$FRAME_COUNT / ($DURATION_MS / 1000.0):.2f}')")
    FRAME_MS=$($PYTHON -c "print(f'{$DURATION_MS / $FRAME_COUNT:.1f}')")
  else
    FPS="0"
    FRAME_MS="0"
  fi

  # Get VRAM usage (nvidia-smi on Jetson)
  VRAM_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

  # Count detections for correctness check
  DETECTION_COUNT=0
  if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then
    DETECTION_COUNT=$(awk '{sum += $3} END {print sum}' "$LOG_FILE" 2>/dev/null || echo "0")
  fi

  echo "  run $i: fps=${FPS} frame_ms=${FRAME_MS}ms duration=${DURATION_MS}ms frames=${FRAME_COUNT} detections=${DETECTION_COUNT} vram=${VRAM_MB}MB"

  # Keep best FPS
  BETTER=$($PYTHON -c "print(1 if float('$FPS') > float('$BEST_FPS') else 0)")
  if [ "$BETTER" = "1" ]; then
    BEST_FPS=$FPS
    BEST_FRAME_MS=$FRAME_MS
    BEST_VRAM=$VRAM_MB
    BEST_DETECTIONS=$DETECTION_COUNT
    BEST_FRAMES=$FRAME_COUNT
    BEST_DURATION=$DURATION_MS
  fi
done

echo ""
echo "METRIC fps=$BEST_FPS"
echo "METRIC frame_ms=$BEST_FRAME_MS"
echo "METRIC duration_ms=$BEST_DURATION"
echo "METRIC vram_mb=$BEST_VRAM"
echo "METRIC detection_count=$BEST_DETECTIONS"
echo "METRIC frame_count=$BEST_FRAMES"
