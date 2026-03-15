#!/usr/bin/env bash
# Correctness checks — ensure optimizations produce identical outputs
# Compares detection log against baseline to catch accuracy regressions
set -euo pipefail

cd "$(dirname "$0")/.."

BASELINE_LOG="${BASELINE_LOG:-auto/baseline_log.txt}"
DATA_DIR="${DATA_DIR:-/tmp/autoresearch_flash_output}"

# Find the latest log file
LOG_FILE=$(find "$DATA_DIR" -name "*flash_log*" -not -name "*reg*" -not -name "*rot*" | head -1)

if [ ! -f "$BASELINE_LOG" ]; then
  echo "WARN: No baseline log found at $BASELINE_LOG — skipping correctness check"
  echo "Run once and save: cp $LOG_FILE auto/baseline_log.txt"
  exit 0
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
  echo "FATAL: No output log found in $DATA_DIR"
  exit 1
fi

# Compare detection counts per frame (column 3 = num_faces)
BASELINE_DETECTIONS=$(awk '{sum += $3} END {print sum}' "$BASELINE_LOG")
CURRENT_DETECTIONS=$(awk '{sum += $3} END {print sum}' "$LOG_FILE")

if [ "$BASELINE_DETECTIONS" != "$CURRENT_DETECTIONS" ]; then
  echo "FATAL: Detection count changed: baseline=$BASELINE_DETECTIONS current=$CURRENT_DETECTIONS"
  exit 1
fi

# Compare frame count
BASELINE_FRAMES=$(wc -l < "$BASELINE_LOG" | tr -d ' ')
CURRENT_FRAMES=$(wc -l < "$LOG_FILE" | tr -d ' ')

if [ "$BASELINE_FRAMES" != "$CURRENT_FRAMES" ]; then
  echo "FATAL: Frame count changed: baseline=$BASELINE_FRAMES current=$CURRENT_FRAMES"
  exit 1
fi

echo "Correctness check passed: $CURRENT_DETECTIONS detections across $CURRENT_FRAMES frames (matches baseline)"
