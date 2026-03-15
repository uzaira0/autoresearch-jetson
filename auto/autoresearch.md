# Autoresearch: FLASH-TV Inference Speed Optimization

## Objective
Optimize the FLASH-TV real-time inference pipeline (face detection → face embedding →
gaze estimation) for maximum throughput on NVIDIA Jetson AGX Orin. Currently ~2 FPS
effective. No accuracy regression allowed — same detections, same outputs, just faster.

## Metrics
- **Primary (optimization target)**: `fps` (frames per second, higher is better)
- **Secondary**:
  - `frame_ms` — per-frame latency through full pipeline (lower is better)
  - `startup_ms` — time from script start to first inference (lower is better)
  - `vram_mb` — peak GPU memory usage (lower is better)
  - `detection_count` — number of faces detected on test video (must not change)
  - `embedding_match` — verification decisions must match baseline exactly

## How to Run
`./auto/autoresearch.sh` — runs the pipeline on a test video, outputs `METRIC name=number` lines.

## Files in Scope
- `python_scripts/test_vid_frames_batch_v7_2fps_frminp_newfv_rotate.py` — **main inference pipeline** (1,265 lines). The primary optimization target.
- `python_scripts/flash_utils.py` — utility functions (distance metrics, camera stream, face drawing)
- `python_scripts/net.py` — AdaFace IR-101 model architecture definition (IR-18/34/50/101/152/200 variants)
- `python_scripts/cv2_capture_automate.py` — video capture automation

## Off Limits
- Model weights/checkpoints — do NOT retrain or swap models without explicit approval
- Output format — log file columns and format must stay identical
- Detection/verification thresholds — do NOT change (0.436 verification, 0.15 detection)
- Gallery structure — reference face format must stay compatible

## Constraints
- All optimizations must produce **identical outputs** on the test video (same detections, same gaze angles to within floating point tolerance)
- Must run on Jetson AGX Orin (ARM64, CUDA, JetPack)
- Python 3.8 compatibility required
- No new pip dependencies unless they provide >20% improvement
- VRAM must stay under 8GB peak

## Hardware
- **Target**: NVIDIA Jetson AGX Orin (64GB or 32GB variant)
- **GPU**: Ampere architecture, CUDA 11.4+, TensorRT 8.x
- **CPU**: 12-core ARM Cortex-A78AE
- **Camera**: Logitech C930e or Anker PowerConf C300 (1920x1080 @ 30 FPS)

## Pipeline Architecture
```
Frame Capture (1080p @ 30fps, throttled to 2fps batches)
    ↓
RetinaFace Detection (608×342, det_thresh=0.15)
    ↓
Face Alignment (InsightFace MXNet, → 112×112)
    ↓
AdaFace IR-101 Embedding (112×112 → 512-dim, batch=7-8)
    ↓
Cosine Distance Matching (threshold=0.436, 6 gallery refs × 2 flips = 12 embeddings)
    ↓
GazeLSTM (7 frames × 224×224, batch=10)
    ↓
Log Output (timestamp, gaze_pitch, gaze_yaw, angular_error, bbox)
```

## Strategic Direction

### Phase 1: Low-hanging fruit (no model changes)
1. **FP16 inference** — `model.half()` on all PyTorch models. Jetson Ampere has native FP16 tensor cores.
2. **torch.no_grad() everywhere** — ensure no gradient computation during inference
3. **CUDA stream pipelining** — overlap detection/embedding/gaze on separate CUDA streams
4. **Batch size tuning** — embedding batch (7-8), gaze batch (10) — sweep for GPU occupancy
5. **Preprocessing on GPU** — move resize/normalize from CPU OpenCV to GPU torchvision transforms
6. **Dead code profiling** — rotation variants, multiple detection paths, unused branches

### Phase 2: Model optimization (same weights)
7. **TorchScript** — `torch.jit.trace()` the GazeLSTM and AdaFace backbone
8. **TensorRT conversion** — convert PyTorch models to TensorRT FP16/INT8 engines
9. **ONNX export** — intermediate step for TensorRT or other runtimes
10. **Operator fusion** — Conv+BN+ReLU fusion in IR-101 backbone

### Phase 3: Architecture exploration (needs accuracy validation)
11. **IR-101 → IR-50** — half the layers, test if verification accuracy holds
12. **Embedding dim 512 → 256** — smaller embeddings, faster distance computation
13. **GazeLSTM temporal window** — 7 frames → 3 or 5, test accuracy tradeoff
14. **RetinaFace → lighter detector** — SCRFD or YOLOv8-face

## Baseline
- **Commit**: (fill in after first run on Jetson)
- **fps**: ~2 (estimated from current 2fps throttle)
- **frame_ms**: (measure)
- **startup_ms**: (measure)
- **vram_mb**: (measure)
- **detection_count**: (measure on test video)

## What's Been Tried
<!-- Update this section as experiments accumulate -->

## Current Best
- **fps**: ~2 (baseline, pre-optimization)
