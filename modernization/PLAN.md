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

## The Cascade Problem

FLASH-TV has a 4-model pipeline where each model's output feeds the next:

```
RetinaFace (detection) → InsightFace (alignment) → AdaFace (embedding) → GazeLSTM (gaze)
         ↓                       ↓                        ↓                      ↓
   face bounding box      112×112 crop              512-dim vector         pitch/yaw angles
```

**Any numerical change at any stage cascades downstream.** A 1-pixel shift in face
alignment changes the crop → changes the embedding → changes the cosine distance →
may cross the 0.436 threshold differently → different identity decision → gaze runs
on wrong person or doesn't run at all.

This means most "same model, different backend" changes are NOT safe without
validation. FP16, TensorRT, ONNX, and even different OpenCV resize interpolation
all introduce floating point differences that can flip threshold decisions.

---

## TRULY SAFE WITHOUT TEST DATA

These changes have **zero effect on model inference**. They change the environment
or non-inference code paths only.

### Phase 0: Infrastructure (zero model risk)

**0.1 — Python 3.8 → 3.11+**
- Pure interpreter upgrade. Same bytecode semantics, same floating point behavior.
- Fix any syntax deprecations (minimal).
- Verify: import all modules, load all models successfully.
- Benefit: ~25% faster CPython interpreter, security patches.
- Risk: None. Python float behavior is identical across versions.
- Effort: 1-2 hours.

**0.2 — PyTorch upgrade to 2.4+**
- Load same checkpoint files into newer PyTorch.
- PyTorch is backward-compatible for model loading and inference.
- Verify: load each model, run forward pass on a single dummy tensor, confirm output shape.
- Benefit: unlocks torch.compile() and better CUDA kernels for later phases.
- Risk: Extremely low. Same CUDA kernels produce same results.
- Effort: 1 hour.

**0.3 — JetPack upgrade to 6.x**
- Flash latest JetPack on the Orin.
- Ships with CUDA 12.x, TensorRT 10.x, cuDNN 9.x.
- Verify: all models load. Forward pass shapes correct.
- Benefit: prerequisite for all later speed optimizations.
- Risk: Low. Same GPU, same model weights.
- Effort: 2-3 hours.

**0.4 — Remove 2fps throttle for benchmarking**
- The capture loop has artificial `sleep()` to limit to 2fps.
- Add a `--benchmark` flag that processes frames as fast as possible.
- Verify: N/A — this only affects timing, not model outputs.
- Risk: Zero. Just removes a sleep.
- Effort: 30 minutes.

**0.5 — Dead code profiling**
- Profile with `cProfile` to identify where wall-clock time is spent.
- Identify unused code paths, unnecessary copies, redundant computation.
- Don't change anything yet — just measure and document.
- Risk: Zero. Read-only analysis.
- Effort: 1-2 hours.

**0.6 — CUDA stream pipelining**
- Run detection/embedding/gaze on separate CUDA streams.
- This changes execution ORDER, not computation. Same ops, same inputs, same outputs.
- Verify: diff log output against single-stream baseline on same input.
- Risk: Very low. Identical math, just overlapped scheduling.
- Effort: 3-4 hours.

**0.7 — Batch size tuning**
- Sweep embedding batch: 4, 8, 16, 32.
- Sweep gaze batch: 5, 10, 20.
- Batch size doesn't change the computation, just how many inputs are processed at once.
- Verify: diff log output per batch size. Must be bit-identical.
- Risk: Very low. Same computation, different grouping.
- Effort: 1 hour per sweep.

---

## NEEDS TEST DATA (any change that touches model numerics)

**Why:** Floating point arithmetic is not associative. Different backends, precisions,
operator implementations, and even batch sizes can produce slightly different results.
In a threshold-based pipeline (cosine distance vs 0.436), "slightly different" can mean
"different identity decision." You MUST validate against ground-truth labels.

### Phase 1: Drop MXNet → ONNX backend

**1.1 — Replace MXNet InsightFace with ONNX backend**
- "Same model weights" does NOT mean identical outputs across MXNet vs ONNX.
- Different BLAS libraries, different operator implementations, different rounding.
- The aligned face crop WILL be slightly different → cascades through entire pipeline.
- **Requires**: labeled frames to verify identity decisions don't change.
- Benefit: eliminates dead MXNet dependency, enables TensorRT acceleration.
- Effort: 3-4 hours.

### Phase 2: Speed optimizations (change model numerics)

**2.1 — FP16 inference**
- `model.half()` reduces precision from 32-bit to 16-bit float.
- Gaze angles WILL change (typically 0.001°-0.1° drift).
- Cosine distances WILL change → threshold decisions may flip.
- **Requires**: labeled frames to verify no identity/gaze regressions.
- Benefit: ~2x on Jetson Ampere tensor cores.
- Effort: 1 hour.

**2.2 — torch.compile()**
- Usually bit-identical but NOT guaranteed. Uses different fused kernels.
- **Requires**: labeled frames to verify (or at minimum, extensive output diffing).
- Benefit: 10-30% speedup.
- Effort: 30 minutes.

**2.3 — TorchScript trace**
- Graph-level optimizations may change operator execution order.
- Floating point results may differ slightly.
- **Requires**: labeled frames to verify.
- Benefit: faster startup, JIT optimizations.
- Effort: 1-2 hours.

**2.4 — TensorRT conversion**
- Completely different inference engine. Different kernels, different precision handling.
- Outputs WILL differ from PyTorch. This is the biggest numerical change.
- **Requires**: labeled frames to verify — this is mandatory.
- Benefit: 2-4x inference speedup (the single biggest win on Jetson).
- Effort: 4-6 hours.

**2.5 — GPU preprocessing pipeline**
- Different resize/normalize implementation (torchvision vs OpenCV).
- Interpolation differences → different pixel values → cascades through models.
- **Requires**: labeled frames to verify.
- Benefit: eliminates CPU→GPU copies.
- Effort: 2 hours.

**2.6 — Dead code removal (rotation variants)**
- Rotation code paths detect faces that upright detection misses.
- Removing them changes detection behavior (fewer detections on rotated faces).
- **Requires**: labeled frames to measure detection recall impact.
- Effort: 2 hours.

### Phase 3: Threshold tuning

**3.1 — Verification threshold sweep**
- Current: hardcoded 0.436 cosine distance.
- Sweep: 0.3 → 0.6 in 0.01 steps.
- Metric: F1 score on labeled identity data.
- **Requires**: labeled frames with known identities.

**3.2 — Detection threshold sweep**
- Current: 0.15 confidence.
- Sweep: 0.05 → 0.5.
- Metric: precision/recall on labeled face bounding boxes.
- **Requires**: frames with annotated face bounding boxes.

**3.3 — Area threshold tuning**
- Current: 35px minimum face area.
- Sweep: 20 → 100.
- **Requires**: labeled frames at various distances.

### Phase 4: Model architecture changes

**4.1 — IR-101 → IR-50 backbone**
- Different model, different accuracy characteristics.
- **Requires**: labeled identity pairs for verification accuracy.

**4.2 — Embedding dimension 512 → 256**
- **Requires**: labeled identity data with hard pairs (siblings).

**4.3 — GazeLSTM temporal window: 7 → 3 or 5 frames**
- **Requires**: frames with ground-truth gaze angles.

**4.4 — Replace RetinaFace with SCRFD**
- **Requires**: annotated face bounding boxes.

**4.5 — Replace GazeLSTM with L2CS-Net or GazeTR**
- **Requires**: labeled gaze data.

**4.6 — End-to-end model distillation**
- **Requires**: large labeled dataset for training.

---

## Recommended Execution Order

```
WITHOUT TEST DATA (do now):
  0.1  Python 3.11         ──┐
  0.2  PyTorch 2.4+         │  Infrastructure (zero model risk)
  0.3  JetPack 6.x          │
  0.4  Remove 2fps throttle │
  0.5  Dead code profiling  │
  0.6  CUDA stream pipeline │
  0.7  Batch size tuning   ──┘

WHEN TEST DATA AVAILABLE (validation required):
  1.1  Drop MXNet → ONNX   ──── Dependency cleanup
  2.1  FP16 inference       ──┐
  2.2  torch.compile()       │
  2.4  TensorRT conversion   │  Speed (autoresearch loop)
  2.5  GPU preprocessing    ──┘
  2.6  Remove rotation       ──── Measure recall impact
  3.*  Threshold sweeps      ──── Accuracy tuning
  4.*  Architecture swaps    ──── Biggest changes last
```

## Expected Impact

| Phase | Expected FPS gain | Requires test data | Risk |
|-------|------------------|--------------------|------|
| Phase 0 (infra) | +25% (Python + batching) | **No** | Zero |
| Phase 1 (drop MXNet) | +10-20% | **Yes** | Medium (cascade) |
| Phase 2 (speed opts) | +200-400% | **Yes** | Medium (numerics) |
| Phase 3 (thresholds) | 0% FPS, better accuracy | **Yes** | Low |
| Phase 4 (architecture) | +50-100% additional | **Yes** | High |

**Without test data (Phase 0 only):** ~2 FPS → ~2.5-3 FPS. Modest, but sets up
the foundation for everything else.

**With test data (Phases 0-2):** ~2 FPS → 10-20 FPS. TensorRT is the big win
but cannot be validated without labeled data.

**The honest answer:** Most of the real speed gains (FP16, TensorRT, ONNX) change
model numerics and need test data to validate. Phase 0 is genuinely safe but gives
you infrastructure, not speed. Get the test data.
