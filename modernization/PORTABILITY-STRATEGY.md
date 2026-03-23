# FLASH-TV Portability Strategy — Beyond Jetson

## Goal

Run the FLASH-TV pipeline on affordable consumer GPU setups (mini-ITX builds) and
cheaper ML-capable edge devices, moving away from Jetson dependency.

Jetsons are expensive, NVIDIA's edge ecosystem is unpredictable, and availability
fluctuates. Consumer hardware is cheaper, more available, and easier to source/replace.

---

## Why This Is Feasible

FLASH-TV is four relatively small models doing a specific task — not an LLM needing
48GB of VRAM. The problem is entirely that nobody has compressed the models yet.
They're running bloated FP32 on a dead framework (MXNet 1.6).

Current memory footprint (FP32, all four models): **~13 GB**
With FP16 + single gaze model: **~5.4 GB**
With INT8 TensorRT + single gaze model: **~2.5-3 GB**

---

## Target Hardware Tiers

### Tier 1: Drop-in replacement (~$250, mini-ITX)

| Hardware | GPU VRAM | Price | Notes |
|----------|----------|-------|-------|
| RTX 4060 (desktop) | 8 GB | ~$250 | Best perf/watt, full CUDA + TensorRT |
| RTX 3050 (desktop) | 8 GB | ~$130 | Full CUDA + TensorRT |

FP16 pipeline (~5.4 GB) fits comfortably. 10+ FPS expected with TensorRT.
Full NVIDIA stack — CUDA, cuDNN, TensorRT all work natively.

### Tier 2: Budget builds (~$80-130, mini-ITX)

| Hardware | GPU VRAM | Price | Notes |
|----------|----------|-------|-------|
| Used GTX 1660 Super | 6 GB | ~$80 | Older but capable, Turing arch |
| Intel Arc A380 | 6 GB | ~$100 | ONNX Runtime has Arc support via DirectML |
| Radeon RX 6500 XT | 4 GB | ~$90 | ROCm support improving, needs INT8 |

FP16 fits on 6 GB cards. INT8 required for 4 GB cards.
Non-NVIDIA cards need ONNX Runtime (no TensorRT).

### Tier 3: Dedicated edge accelerators (~$70-100)

| Hardware | Memory | Price | Notes |
|----------|--------|-------|-------|
| Coral TPU (M.2/USB) | N/A | ~$70 | INT8 only, TFLite models |
| Hailo-8 (M.2) | N/A | ~$100 | INT8 only, needs full quantization |

Requires complete INT8 quantization and model format conversion.
Highest effort but lowest power draw and cost.

---

## What This Changes About Optimization Priorities

### 1. Kill MXNet first, not last

MXNet is the single biggest portability blocker. It's dead software (archived by Apache
2023) that only compiles easily on Jetson because NVIDIA shipped it pre-built. On a
consumer desktop you'd be building MXNet 1.6 from source against modern CUDA — nightmare.

**Action:** Convert RetinaFace to ONNX and MXNet is gone forever.

### 2. ONNX becomes the target runtime, not TensorRT

TensorRT is NVIDIA-only. ONNX Runtime runs on:
- **CUDA** (NVIDIA GPUs)
- **TensorRT EP** (NVIDIA GPUs, via ONNX Runtime — best of both worlds)
- **DirectML** (Intel Arc, AMD, any DirectX 12 GPU on Windows)
- **ROCm** (AMD GPUs on Linux)
- **OpenVINO** (Intel CPUs and iGPUs)
- **CPU** (fallback, any x86/ARM)

If you build around ONNX Runtime, you get every hardware target for free. You can still
use TensorRT as an ONNX Runtime execution provider for NVIDIA GPUs — no lock-in.

### 3. Model compression is non-negotiable

On a 64GB AGX Orin you can postpone FP16/INT8 because memory isn't the constraint.
For a $130 mini-ITX build with 8GB VRAM, you need the models small. This is the core work.

### 4. Smaller backbones become more attractive

- AdaFace: IR-101 → IR-50 (or even IR-18)
- Gaze: ResNet50 → ResNet18 or MobileNetV3
- Detection: RetinaFace → SCRFD or YOLOv8-face

On a Jetson AGX with 16 SMs the bigger model is fine. On a GTX 1660 with 22 SMs but
6GB VRAM, a smaller model at INT8 will actually be faster.

### 5. Python 3.8 upgrade matters for different reasons

Modern `onnxruntime` wheels have dropped Python 3.8 support. You need 3.10+ to get
current ONNX Runtime with all execution providers.

---

## Revised Priority Order (Portability-First)

```
1. ONNX export all models
   - RetinaFace (eliminates MXNet forever)
   - AdaFace IR-101
   - GazeLSTM (primary only, drop secondary)

2. ONNX Runtime inference
   - Replace PyTorch + MXNet runtime with onnxruntime-gpu
   - Use CUDAExecutionProvider on NVIDIA, DirectML on Intel/AMD

3. FP16 ONNX models
   - Halves memory, runs everywhere with GPU support
   - Pipeline fits in ~5.4 GB → any 6-8 GB GPU works

4. Python 3.10+
   - Needed for modern onnxruntime wheels
   - Unlocks latest execution providers

5. INT8 quantization (with calibration data)
   - Gets pipeline to ~2.5-3 GB → 4 GB GPUs viable
   - Required for Coral TPU / Hailo-8 edge accelerators

6. Smaller backbones
   - IR-101 → IR-50 or IR-18
   - ResNet50 gaze → ResNet18 or MobileNet
   - Gets to ~1.5-2 GB → integrated GPUs viable

7. TensorRT as optional accelerator
   - Via ONNX Runtime's TensorRT EP, not a hard dependency
   - Automatically used when available, graceful fallback when not
```

---

## The Cascade Problem Still Applies

Every step from 1 onward changes model numerics and can flip the 0.436 cosine
threshold. The validation strategy:

- **Jetson is the test rig** — validate accuracy on Jetson with labeled data
- **Deploy anywhere** — once validated, the ONNX model is portable
- Same model file, same weights, same behavior across all ONNX Runtime backends
  (within floating point tolerance — which is why you validate once)

---

## Expected Outcome

| Configuration | Memory | Min GPU | Est. Cost | Est. FPS |
|--------------|--------|---------|-----------|----------|
| Current (FP32, all models, MXNet) | ~13 GB | Jetson AGX 32GB | $800+ | ~2 |
| FP16, single gaze, ONNX | ~5.4 GB | RTX 3050 8GB | ~$130 | 10-15 |
| INT8, single gaze, ONNX | ~2.5 GB | GTX 1650 4GB | ~$80 | 15-25 |
| INT8, smaller backbones, ONNX | ~1.5 GB | Intel Arc A380 | ~$100 | 8-15 |

The pipeline is small enough that a well-optimized version (ONNX + FP16 + single gaze
model) should run at 10+ FPS on a $130 GPU. The key insight: **optimize for ONNX
portability, not Jetson-specific tricks.** The Jetson becomes the development/validation
platform, not the deployment target.
