# Autoresearch Agent Instructions — FLASH-TV

You are running an autonomous experiment loop to optimize FLASH-TV inference speed
on NVIDIA Jetson AGX Orin.

## The Loop

```
read auto/autoresearch.md → understand pipeline + what's been tried
    ↓
form hypothesis → pick ONE focused change
    ↓
edit code → make the change (small, surgical)
    ↓
git commit → commit with descriptive message
    ↓
./auto/autoresearch.sh → run benchmark
    ↓
evaluate result:
  fps improved AND detection_count matches baseline → keep
  fps worse/equal OR detection_count changed → discard
  crash → fix if trivial, else discard
    ↓
repeat forever
```

## Rules

1. **LOOP FOREVER.** Never ask "should I continue?"
2. **FPS is king.** Higher FPS → keep. Lower or equal → discard.
3. **Detection count must not change.** Same frames in → same detections out.
4. **One change at a time.** Each experiment = single hypothesis.
5. **Commit before benchmarking.** Include result in commit message.
6. **Update autoresearch.md** after every 5-10 experiments.
7. **Profile before guessing.** Use `torch.cuda.Event` timers or `cProfile` to find actual bottlenecks. Don't optimize code that isn't slow.
8. **Phase 1 first.** Exhaust no-model-change optimizations before touching model architecture.
9. **Test on the SAME video every time.** Consistency is key for valid comparisons.

## Key Files
- `python_scripts/test_vid_frames_batch_v7_2fps_frminp_newfv_rotate.py` — main pipeline (1,265 lines)
- `python_scripts/net.py` — model architecture
- `python_scripts/flash_utils.py` — utilities

## NEVER STOP. Keep going until interrupted.
