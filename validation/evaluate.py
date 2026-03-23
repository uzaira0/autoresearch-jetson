"""Core evaluation logic: pipeline logs + VATIC → epoch-based comparison.

This module implements the same methodology as the original Evaluation/
scripts but as a clean, reusable pipeline:

    1. Parse VATIC annotations → 1/sec gaze time series
    2. Parse pipeline _reg.txt + _rot.txt → 1/sec gaze predictions
    3. Align both by timestamp
    4. Condense to 5-second epochs (majority vote)
    5. Confusion matrix
"""

import numpy as np
import pandas as pd

from .gaze import classify_gaze, correct_rotation, load_gaze_limits
from .parsers import parse_pipeline_log, vatic_to_timeseries
from .types import EpochResult, FamilyInput


def pipeline_to_timeseries(
    reg_file: str,
    rot_file: str,
    gaze_limits_file: str,
    tv_config: "TVConfig",
) -> pd.DataFrame:
    """Process pipeline _reg.txt and _rot.txt into 1/sec gaze predictions.

    Replicates the logic from Evaluation/logdata.py:
    - Apply rotation correction for |head_angle| >= 30°
    - Apply 60/40 mix model (rot gaze + reg gaze)
    - Classify gaze using spatial grid thresholds
    - Fill into 1-per-second DataFrame
    - Each prediction covers 2 seconds (pipeline captures at ~2fps)

    Returns DataFrame indexed by timestamp with columns:
        'TC_gaze': 1=gaze, 0=no-gaze
        'TC_exposure_only': 1=TC-present-not-gazing, 0=other
    """
    reg_df = parse_pipeline_log(reg_file)
    rot_df = parse_pipeline_log(rot_file)

    # Build second-level index spanning the full time range
    start = min(reg_df.index[0], rot_df.index[0])
    end = max(reg_df.index[-1], rot_df.index[-1])
    gz_index = pd.date_range(start=str(start), end=str(end), freq="s")
    gz_df = pd.DataFrame(index=gz_index)
    gz_df["TC_gaze"] = 5  # 5 = unset
    gz_df["TC_exposure_only"] = 5

    # Apply rotation correction on reg data where head angle >= 30°
    rotated = reg_df[reg_df["rot."].abs() >= 30]
    if len(rotated) > 0:
        phi_data = rotated[["phi", "theta", "rot."]].values.astype(float)
        corrected = correct_rotation(phi_data)
        reg_df.loc[rotated.index, ["phi", "theta"]] = corrected[:, :2]

    # No-detection frames: TC not present
    no_det = rot_df[rot_df["tag"] == "Gaze-no-det"]
    gz_df.loc[no_det.index, "TC_gaze"] = 0
    gz_df.loc[no_det.index, "TC_exposure_only"] = 0
    # Each prediction covers 2 seconds
    inc_index = no_det.index + pd.Timedelta(seconds=1)
    inc_index = inc_index[inc_index <= gz_df.index.max()]
    gz_df.loc[inc_index, "TC_gaze"] = 0
    gz_df.loc[inc_index, "TC_exposure_only"] = 0

    # Gaze-detected frames: classify gaze direction
    gaze_det = rot_df[rot_df["tag"] == "Gaze-det"]
    gaze_det_reg = reg_df.loc[gaze_det.index]

    if len(gaze_det) > 0:
        gz_data = gaze_det[["phi", "theta", "top", "left", "bottom", "right"]].values.copy()
        gz_reg = gaze_det_reg[["phi", "theta"]].values.copy()

        # Ensure float type
        gz_data = gz_data.astype(float)
        gz_reg = gz_reg.astype(float)

        # Mix model: 60% rotation-corrected + 40% regression
        gz_data[:, :2] = 0.6 * gz_data[:, :2] + 0.4 * gz_reg

        # Load gaze limits and classify
        gaze_limits = load_gaze_limits(gaze_limits_file, tv_config)
        pred_gz = classify_gaze(gz_data, gaze_limits).astype(np.int32)
        tc_exp_only = 1 - pred_gz

        gz_df.loc[gaze_det.index, "TC_gaze"] = pred_gz
        inc_index = gaze_det.index + pd.Timedelta(seconds=1)
        inc_index = inc_index[inc_index <= gz_df.index.max()]
        gz_df.loc[inc_index[:len(pred_gz)], "TC_gaze"] = pred_gz[:len(inc_index)]

        gz_df.loc[gaze_det.index, "TC_exposure_only"] = tc_exp_only
        gz_df.loc[inc_index[:len(tc_exp_only)], "TC_exposure_only"] = tc_exp_only[:len(inc_index)]

    # Fill remaining unset values as 0
    gz_df[gz_df == 5] = 0

    return gz_df


def epoch_vote(arr: np.ndarray) -> int:
    """Majority vote over a 5-element array. Requires ≥3 agreement."""
    votes = [(arr == v).sum() for v in range(4)]
    idx = int(np.argmax(votes))
    return idx if votes[idx] >= 3 else 0


def condense_to_epochs(series: np.ndarray) -> np.ndarray:
    """Condense 1/sec data to 5-second epochs via majority vote."""
    trim = series.size % 5
    if trim:
        series = series[:-trim]
    return np.apply_along_axis(epoch_vote, axis=1, arr=series.reshape(-1, 5))


def _majority(values):
    """Pick the most common value in a group."""
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def _resample_to_5sec(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Resample a 1/sec DataFrame to 5-second epochs on clock boundaries.

    Aligns to :00, :05, :10, ... second boundaries so that independently
    timed series land on the same grid.
    """
    # Find the first timestamp whose second is divisible by 5
    aligned_start = df.index[0]
    offset = aligned_start.second % 5
    if offset != 0:
        aligned_start = aligned_start + pd.Timedelta(seconds=5 - offset)

    # Bin into 5-second windows
    bins = pd.date_range(start=aligned_start, end=df.index[-1], freq="5s")
    if len(bins) < 2:
        return pd.DataFrame(columns=[col])

    cut = pd.cut(df.index, bins=bins, right=False)
    grouped = df.groupby(cut, observed=False)[col].agg(_majority)

    result = pd.DataFrame({col: grouped.values}, index=bins[:-1])
    return result


def evaluate_family(family: FamilyInput) -> EpochResult:
    """Run full epoch-based evaluation for one family.

    This is the main entry point. Provide a FamilyInput and get back
    an EpochResult with confusion matrix and metrics.
    """
    # Step 1: VATIC → 1/sec gaze time series
    vatic_ts = vatic_to_timeseries(
        family.vatic_file,
        family.video_start_time,
        fps=family.fps,
        end_frame=family.end_frame,
    )

    # Step 2: Pipeline → 1/sec gaze predictions
    pipeline_ts = pipeline_to_timeseries(
        family.pipeline_reg_file,
        family.pipeline_rot_file,
        family.gaze_limits_file,
        family.tv_config,
    )

    # Step 3: Resample both to 5-second epochs aligned to clock boundaries.
    #
    # The original cm_vatic.py aligns epochs to the nearest :00/:05/:10/...
    # boundary, NOT to each series' start time. This ensures the two
    # independently-started series land on the same grid.
    pipeline_ts_5 = _resample_to_5sec(pipeline_ts, "TC_gaze")
    vatic_ts_5 = _resample_to_5sec(vatic_ts, "gaze")

    # Step 4: Merge on aligned timestamps
    merged = pipeline_ts_5.join(vatic_ts_5, how="inner")
    merged = merged.dropna()

    # Rename for clarity
    merged = merged.rename(columns={"TC_gaze": "pipeline", "gaze": "vatic"})

    # Filter to valid gaze values (0 or 1 only)
    merged = merged[(merged["pipeline"].isin([0, 1])) & (merged["vatic"].isin([0, 1]))]

    # Step 5: Confusion matrix
    pl = merged["pipeline"].values.astype(int)
    gt = merged["vatic"].values.astype(int)

    tp = int(((pl == 1) & (gt == 1)).sum())
    tn = int(((pl == 0) & (gt == 0)).sum())
    fp = int(((pl == 1) & (gt == 0)).sum())
    fn = int(((pl == 0) & (gt == 1)).sum())
    total = tp + tn + fp + fn

    return EpochResult(
        family_id=family.family_id,
        total_epochs=len(merged),
        gaze_epochs=total,
        true_pos=tp,
        true_neg=tn,
        false_pos=fp,
        false_neg=fn,
        sensitivity=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        accuracy=(tp + tn) / total if total else 0.0,
        pipeline_gaze_seconds=float((pl == 1).sum() * 5),
        vatic_gaze_seconds=float((gt == 1).sum() * 5),
    )
