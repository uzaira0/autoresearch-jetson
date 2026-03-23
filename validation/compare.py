"""Frame-level diff between two pipeline runs (no ground truth needed)."""

from typing import Dict

from .parsers import parse_flash_log_raw
from .types import DiffResult, PipelineFrame


def _gaze_to_bool(pf: PipelineFrame):
    status = (pf.status or "").lower()
    if "gaze-det" in status and "no" not in status:
        return True
    if "gaze-no-det" in status or "no-gaze" in status:
        return False
    return None


def diff_pipeline_runs(
    baseline: Dict[int, PipelineFrame],
    optimized: Dict[int, PipelineFrame],
) -> DiffResult:
    """Compare two pipeline runs frame-by-frame.

    Useful for verifying that an optimization produces identical
    (or near-identical) results to the baseline.
    """
    all_frames = sorted(set(baseline.keys()) | set(optimized.keys()))
    matched = sorted(set(baseline.keys()) & set(optimized.keys()))

    tc_agree = tc_disagree = det_differ = 0
    gaze_agree = gaze_disagree = 0
    iou_sum = 0.0
    bbox_frames = 0

    for f in matched:
        b, o = baseline[f], optimized[f]

        if b.tc_identified == o.tc_identified:
            tc_agree += 1
        else:
            tc_disagree += 1

        if b.num_faces != o.num_faces:
            det_differ += 1

        bg, og = _gaze_to_bool(b), _gaze_to_bool(o)
        if bg is not None and og is not None:
            if bg == og:
                gaze_agree += 1
            else:
                gaze_disagree += 1

        if b.tc_bbox and o.tc_bbox:
            iou_sum += b.tc_bbox.iou(o.tc_bbox)
            bbox_frames += 1

    return DiffResult(
        total_frames=len(all_frames),
        matched_frames=len(matched),
        tc_id_agree=tc_agree,
        tc_id_disagree=tc_disagree,
        detection_count_differ=det_differ,
        gaze_agree=gaze_agree,
        gaze_disagree=gaze_disagree,
        mean_bbox_iou=iou_sum / bbox_frames if bbox_frames else None,
        bbox_frames=bbox_frames,
    )
