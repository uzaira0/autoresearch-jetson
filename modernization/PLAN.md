# FLASH-TV Modernization Plan

## Current State (outdated)

| Component | Current | Status | Risk |
|-----------|---------|--------|------|
| Python | 3.8 | EOL Oct 2024 | Security + performance |
| MXNet | 1.6.0 | Archived by Apache 2023 | Dead dependency, no updates |
| PyTorch | <2.0 | Missing torch.compile, FP16 optimizations | Performance |
| InsightFace | MXNet backend | Upstream switched to ONNX years ago | Stuck on dead backend |
| CUDA/TensorRT | Older JetPack | Missing latest Jetson optimizations | Performance |
| RetinaFace | Original implementation | Works but slower than modern alternatives | Performance |
| GazeLSTM | Custom LSTM + ResNet-50 | Works but no modern gaze architectures | Accuracy ceiling |

---

## NO TEST DATA REQUIRED

These changes are **mechanically verifiable** — same inputs produce same outputs,
or the change is purely infrastructure with no model behavior change.

### Phase 0: Infrastructure (zero model risk)

**0.1 — Python 3.8 → 3.11+**
- Update venv and all pip packages
- Fix any syntax/API deprecations (minimal — Python is backward-compatible)
- Verify: `python -c "import torch; import cv2; print('ok')"` + run pipeline, diff log output
- Benefit: ~25% faster CPython interpreter, security patches
- Effort: 1-2 hours

**0.2 — PyTorch upgrade to 2.4+**
- Install latest PyTorch wheel for Jetson (NVIDIA provides ARM builds)
- No code changes needed — PyTorch is backward-compatible
- Verify: load all model checkpoints, run inference on 1 frame, compare outputs
- Benefit: unlocks `torch.compile()`, better CUDA kernels, native FP16 path
- Effort: 1 hour (just pip install + verify)

**0.3 — JetPack upgrade to 6.x**
- Flash latest JetPack on the Orin
- Ships with CUDA 12.x, TensorRT 10.x, cuDNN 9.x
- Verify: all models load and produce same outputs
- Effort: 2-3 hours (flash + reinstall Python environment)

### Phase 1: Drop MXNet (mechanical replacement)

**1.1 — Replace MXNet InsightFace with ONNX backend**
- InsightFace switched from MXNet to ONNX years ago
- The `model-r100-ii` face alignment model has an ONNX equivalent in the InsightFace model zoo
- Replace `FaceModelv2` (MXNet) with `insightface.app.FaceAnalysis` (ONNX)
- Verify: extract aligned faces from 100 test frames, pixel-diff against MXNet output (should be identical or negligible floating point diff)
- Benefit: eliminates MXNet dependency entirely, ONNX runs faster via TensorRT
- Effort: 3-4 hours
- **NO test data needed** — this is a backend swap, same model weights, same alignment

**1.2 — Remove all MXNet imports and dependencies**
- After 1.1 verified, strip MXNet from requirements, remove dead code paths
- Effort: 30 minutes

### Phase 2: Pure speed optimizations (same outputs)

**2.1 — FP16 inference**
- `model.half()` on AdaFace IR-101 and GazeLSTM
- `input_tensor = input_tensor.half()` for all model inputs
- Verify: compare log outputs — gaze angles within 0.01° tolerance, same detections
- Benefit: ~2x on Jetson Ampere tensor cores
- Effort: 1 hour
- **NO test data needed** — numerical outputs compared directly

**2.2 — torch.compile() on hot models**
- `compiled_model = torch.compile(model, mode="reduce-overhead")`
- Apply to AdaFace backbone and GazeLSTM
- Verify: same outputs, measure FPS improvement
- Benefit: 10-30% on PyTorch 2.x
- Effort: 30 minutes
- **NO test data needed**

**2.3 — TorchScript trace**
- `traced = torch.jit.trace(model, sample_input)` for all PyTorch models
- Save traced models to disk for faster subsequent loads
- Verify: same outputs on sample frames
- Benefit: faster startup, JIT optimizations
- Effort: 1-2 hours
- **NO test data needed**

**2.4 — TensorRT conversion**
- Export PyTorch models → ONNX → TensorRT FP16 engines
- Apply to: RetinaFace, AdaFace IR-101, GazeLSTM
- Verify: same detections, same gaze angles (within tolerance)
- Benefit: 2-4x inference speedup (the single biggest win on Jetson)
- Effort: 4-6 hours (each model needs calibration validation)
- **NO test data needed** — compare outputs frame-by-frame against PyTorch baseline

**2.5 — GPU preprocessing pipeline**
- Replace CPU OpenCV `resize`/`normalize` with `torchvision.transforms` on GPU
- Eliminates CPU→GPU memory copies for preprocessing
- Verify: same pixel values after transform
- Effort: 2 hours
- **NO test data needed**

**2.6 — CUDA stream pipelining**
- Run detection/embedding/gaze on separate CUDA streams
- Overlap GPU compute with CPU work on next frame
- Verify: same outputs (just changes execution order, not compute)
- Effort: 3-4 hours
- **NO test data needed**

**2.7 — Batch size tuning**
- Sweep embedding batch: 4, 8, 16, 32
- Sweep gaze batch: 5, 10, 20
- Autoresearch loop: try each, keep what improves FPS
- Verify: same outputs regardless of batch size
- Effort: 1 hour per sweep
- **NO test data needed**

**2.8 — Remove 2fps throttle for benchmarking**
- The capture loop has artificial sleep/throttle to 2fps
- Remove for benchmark mode to measure true pipeline capacity
- Add `--benchmark` flag that processes all frames as fast as possible
- Effort: 30 minutes
- **NO test data needed**

**2.9 — Dead code profiling and removal**
- Profile with `cProfile` or `torch.cuda.Event` timers
- Rotation variants: measure how often they find faces that upright misses
- If <1% benefit, remove rotation code paths
- Verify: compare detection counts with/without rotation
- Effort: 2 hours
- **NO test data needed** — measure detection count delta directly

---

## TEST DATA REQUIRED

These changes **alter model behavior** — they may detect different faces, produce
different gaze angles, or change verification decisions. Validation against
ground-truth labels is mandatory.

### Phase 3: Threshold tuning (needs labeled data)

**3.1 — Verification threshold sweep**
- Current: hardcoded 0.436 cosine distance
- Sweep: 0.3 → 0.6 in 0.01 steps
- Metric: F1 score on labeled identity data (true positive ID matches)
- Autoresearch target: maximize F1 while keeping FPS constant
- **Requires**: labeled frames with known identities (who is in each frame)

**3.2 — Detection threshold sweep**
- Current: 0.15 confidence
- Sweep: 0.05 → 0.5
- Metric: precision/recall on labeled face bounding boxes
- **Requires**: frames with annotated face bounding boxes

**3.3 — Area threshold tuning**
- Current: 35px minimum face area
- Sweep: 20 → 100
- Metric: detection recall at various distances
- **Requires**: labeled frames with faces at known distances

### Phase 4: Model architecture changes (needs labeled data)

**4.1 — IR-101 → IR-50 backbone**
- Half the layers in the face embedding model
- Faster inference but potentially lower verification accuracy
- Metric: verification F1 score must stay ≥ baseline
- **Requires**: labeled identity pairs for verification accuracy
- Benefit: ~40% faster embedding inference
- Effort: 2-3 hours (swap backbone, load IR-50 pretrained weights, benchmark)

**4.2 — Embedding dimension 512 → 256**
- Smaller embeddings = faster distance computation + less memory
- May lose discriminative power for similar-looking faces
- Metric: verification F1 at various thresholds
- **Requires**: labeled identity data with known hard pairs (siblings)
- Effort: 3-4 hours (retrain or find pre-trained 256-dim model)

**4.3 — GazeLSTM temporal window: 7 → 3 or 5 frames**
- Fewer frames per gaze prediction = faster + lower latency
- May lose temporal smoothing / accuracy
- Metric: gaze angular error on labeled gaze data
- **Requires**: frames with ground-truth gaze angles
- Effort: 2 hours (modify input pipeline, benchmark)

**4.4 — Replace RetinaFace with SCRFD**
- SCRFD is from the same team (InsightFace), 2x faster, better accuracy
- Metric: detection mAP on labeled face data
- **Requires**: annotated face bounding boxes for validation
- Benefit: 2x detection speed with equal or better accuracy
- Effort: 4-6 hours (swap detector, validate outputs)

**4.5 — Replace GazeLSTM with L2CS-Net or GazeTR**
- Modern gaze estimation architectures
- L2CS-Net: ResNet-50 + binned classification (simpler, faster)
- GazeTR: transformer-based (potentially more accurate)
- Metric: gaze angular error
- **Requires**: labeled gaze data
- Effort: 1-2 days (new model integration, benchmark)

**4.6 — End-to-end model distillation**
- Train a single smaller model that mimics the full 4-model pipeline
- Input: raw frame → Output: identity + gaze
- Metric: all metrics must match or exceed the pipeline
- **Requires**: large labeled dataset for training
- Effort: 1-2 weeks

---

## Recommended Execution Order

```
WITHOUT TEST DATA (do now):
  0.1  Python 3.11        ──┐
  0.2  PyTorch 2.4+        │  Infrastructure
  0.3  JetPack 6.x        ──┘
  1.1  Drop MXNet → ONNX  ──── Biggest dependency win
  2.1  FP16 inference      ──┐
  2.2  torch.compile()      │
  2.4  TensorRT conversion  │  Speed (autoresearch loop)
  2.8  Remove 2fps throttle │
  2.5  GPU preprocessing   ──┘

WITH TEST DATA (do when available):
  3.1  Threshold sweep     ──── Quick wins
  4.4  SCRFD detector      ──┐
  4.1  IR-50 backbone       │  Architecture (autoresearch loop)
  4.3  Temporal window      │
  4.5  Modern gaze model   ──┘
```

## Expected Impact

| Phase | Expected FPS gain | Requires test data |
|-------|------------------|--------------------|
| Phase 0 (infra) | +25% (Python speedup) | No |
| Phase 1 (drop MXNet) | +10-20% (ONNX faster) | No |
| Phase 2 (speed opts) | +200-400% (TensorRT + FP16) | No |
| Phase 3 (thresholds) | 0% FPS, better accuracy | Yes |
| Phase 4 (architecture) | +50-100% additional | Yes |

**Conservative estimate**: Phases 0-2 alone should take the pipeline from ~2 FPS to ~10-15 FPS without any accuracy risk. With TensorRT, potentially 20+ FPS.
