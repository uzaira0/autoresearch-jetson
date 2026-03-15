# Autoresearch Ideas — FLASH-TV Inference Speed

## Key Architecture Facts
- 4 models in series: RetinaFace → InsightFace align → AdaFace IR-101 → GazeLSTM
- Jetson AGX Orin: Ampere GPU, 12-core ARM, native FP16 tensor cores, TensorRT 8.x
- Current throughput: ~2 FPS (includes 2fps throttle in capture loop)
- Pipeline is Python-heavy with CPU↔GPU transfers at each model boundary
- Main script: 1,265 lines with rotation variants and multiple detection code paths

## Phase 1: No model changes (pure speed)

### High-confidence
- **FP16 inference**: `model.half()` + `input.half()` on AdaFace and GazeLSTM. Jetson Ampere has dedicated FP16 tensor cores — should be ~2x on matrix ops for free
- **torch.no_grad() audit**: Verify ALL inference paths are wrapped. Missing this = 2x memory + backward graph overhead
- **Remove 2fps throttle**: The capture loop artificially limits to 2fps. For benchmarking, remove the sleep/throttle to measure true pipeline capacity
- **Pre-allocate tensors**: Reuse input tensors across frames instead of allocating new ones per batch
- **Pin memory for CPU→GPU transfers**: `tensor.pin_memory()` enables async DMA copies

### Medium-confidence
- **CUDA streams**: Run detection on stream 1, embedding on stream 2, gaze on stream 3. Overlap GPU compute with CPU preprocessing of next frame
- **Batch size sweep**: Embedding batch=7-8, gaze batch=10. Profile GPU occupancy — might be under-utilizing with small batches
- **GPU preprocessing**: Replace CPU OpenCV resize+normalize with torchvision.transforms on GPU tensors. Eliminates CPU→GPU copy for preprocessing
- **Eliminate rotation variants**: Main script runs detection on rotated frames. Profile how often rotations actually find faces that upright misses — may be wasted compute
- **Reduce detection resolution**: Currently 608×342. Test 480×270 or 320×180 — RetinaFace may still detect faces at lower res

### Speculative
- **Fuse all models into single TensorRT engine**: Detection → embedding → gaze as one optimized graph
- **Replace Python frame capture with C++ GStreamer**: Eliminate Python GIL for capture thread
- **INT8 quantization with TensorRT**: Needs calibration data but could be 2-4x over FP32

## Phase 2: Model optimization (same weights, different format)

- **TorchScript trace**: `torch.jit.trace(model, sample_input)` — enables JIT optimizations
- **TensorRT conversion**: Export to ONNX → TensorRT FP16 engine. Most impactful on Jetson.
- **ONNX Runtime**: May be faster than raw PyTorch for inference on ARM
- **Conv+BN fusion**: IR-101 has 101 Conv→BN→ReLU sequences. Fusing saves memory bandwidth

## Phase 3: Architecture (needs accuracy validation with labeled data)

- **IR-101 → IR-50**: Half the backbone layers. Needs verification accuracy check
- **GazeLSTM temporal window**: 7 → 3 frames. Needs gaze angular error check
- **SCRFD or YOLOv8-face**: Lighter detectors than RetinaFace
- **Embedding dim 512 → 256**: Halves distance computation. Needs re-verification of threshold

## Dead Ends (track here)
<!-- Move ideas here when they don't work -->
