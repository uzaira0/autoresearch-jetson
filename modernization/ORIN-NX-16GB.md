# FLASH-TV on 16GB Jetson Orin NX — Assessment & Optimization Strategy

## Hardware Profile (This Device)

| Spec | Value |
|------|-------|
| Device | NVIDIA Orin NX Developer Kit (Seeed reComputer) |
| JetPack | 5.1 (R35.5.0) |
| CUDA | 11.4 (V11.4.315) |
| cuDNN | 8.6.0 |
| TensorRT | 8.5.2.2 |
| PyTorch | 1.14.0a0+44dac51c.nv23.01 |
| MXNet | 1.6.0 (local build) |
| Python | 3.8 |
| GPU | "Orin" — 8 SMs / 1024 CUDA cores, Ampere compute 8.7 |
| Memory | 15.5 GB unified (CPU+GPU shared) |
| Storage | 116 GB NVMe SSD (83 GB free) |
| Thermals | ~55-58°C under light load, well within limits |

Key differences from AGX Orin:
- **Half the SMs** (8 vs 16)
- **Quarter to sixth the memory** (16GB vs 32-64GB)
- **Unified memory** — GPU allocation directly reduces CPU/OS available memory

---

## The Core Problem: The Pipeline Doesn't Fit As-Is

The `gpu_memory_manager_v2.py` (in flash-tv-scripts, gpu-memory-management branch)
allocates **14GB to GPU** with this breakdown:

| Model | Memory | Framework |
|-------|--------|-----------|
| RetinaFace | 1.5 GB | MXNet |
| AdaFace IR-101 | 5.5 GB | PyTorch |
| GazeLSTM primary | 3.5 GB | PyTorch |
| GazeLSTM secondary | 2.5 GB | PyTorch |
| System buffer | 1.0 GB | — |
| **Total** | **14.0 GB** | |

On this Orin NX, GPU memory IS system memory (unified architecture). Reserving 14GB
for GPU leaves only **~1.5GB for the entire OS, Python runtime, OpenCV, video capture,
and CPU-side processing.** The system already uses ~3.5GB at idle. This is not viable.

All models run **full FP32 with zero quantization**. No TensorRT optimization. No ONNX.
No model compression of any kind.

---

## What the AGX Orin Work Tells Us

The autoresearch-jetson repo (main branch) has solid scaffolding (benchmark harness,
agent loop, modernization plan) but **no actual optimization work has been done yet** —
it's all planning docs. There are no separate branches for different hardware variants.

The modernization plan (PLAN.md) correctly identifies the cascade risk (threshold-sensitive
pipeline) but targets 32/64GB AGX Orin where memory isn't the bottleneck. **None of the
Phase 0 work directly solves the 16GB memory problem.**

---

## Optimization Strategy for 16GB Orin NX

### Priority 1: Drop the second gaze model (saves 2.5GB immediately)

The pipeline loads **two** gaze models (GazeLSTM + GazeLSTMreg). This is a luxury the
16GB device can't afford. On the AGX Orin with 32-64GB, sure. Here, drop the secondary:

- Budget goes from 14GB → 11.5GB for models
- Leaves ~4GB for OS/CPU — much healthier
- Minimal accuracy impact (primary model handles the core task)

### Priority 2: FP16 inference (halves model memory)

All models run full FP32. The Orin NX has Ampere tensor cores (compute 8.7) that
natively support FP16. Converting to FP16:

| Model | FP32 | FP16 (est.) |
|-------|------|-------------|
| RetinaFace (MXNet) | 1.5 GB | ~0.8 GB |
| AdaFace IR-101 | 5.5 GB | ~2.8 GB |
| GazeLSTM primary | 3.5 GB | ~1.8 GB |
| **Total** | **10.5 GB** | **~5.4 GB** |

**Caveat:** The modernization plan correctly warns this changes numerical outputs and can
flip the 0.436 cosine threshold. Validation data is needed. But on 16GB, there may be no
choice.

### Priority 3: TensorRT conversion

TensorRT 8.5.2 is already installed. Converting the PyTorch models (AdaFace + GazeLSTM)
to TensorRT engines would:
- Reduce memory footprint further (optimized graph)
- Give 2-4x inference speedup
- Enable INT8 if calibration data is available

The MXNet RetinaFace should be converted to ONNX first, then TensorRT.

### Priority 4: Reduce the memory manager limit

Change `TOTAL_GPU_MEMORY_LIMIT` from 14.0 to **10.0-11.0 GB** in
`gpu_memory_manager_v2.py`. This leaves 4.5-5.5GB for OS/CPU, which is sustainable.
With FP16 models fitting in ~5.4GB total, this is achievable.

### Priority 5: Dynamic model loading/unloading

If memory is still tight, implement model swapping:
- Load RetinaFace + AdaFace (detection + verification) — ~3.6GB FP16
- Only load GazeLSTM when target child is detected
- Unload when not needed for N seconds

This adds latency on first gaze estimation but keeps steady-state memory low.

---

## What NOT to Do

1. **Don't touch thresholds** (0.436 verification, 0.15 detection) without labeled test data
2. **Don't upgrade JetPack yet** — JetPack 5.1 → 6.x is a full reflash and the Seeed
   reComputer image may not have a JP6 equivalent ready
3. **Don't try to run the same config as the 32/64GB AGX Orin** — it physically won't fit

---

## Recommended Execution Order (16GB-specific)

```
IMMEDIATE (today):
  1. Drop secondary gaze model              saves 2.5 GB
  2. Lower memory limit to 11 GB            healthier OS headroom
  3. Profile actual memory with tegrastats   understand real vs theoretical

NEEDS VALIDATION DATA:
  4. FP16 conversion (AdaFace + GazeLSTM)   halves model memory
  5. ONNX export of RetinaFace              eliminates MXNet
  6. TensorRT conversion of all models      2-4x speedup
  7. Batch size tuning (try smaller)         currently fixed at 8/7
```

Steps 1-3 can be done today. Steps 4-7 need test data for validation but are required
to hit reasonable FPS on this hardware.
