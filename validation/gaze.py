"""Gaze estimation logic.

Ported from Evaluation/gaze_estimate.py. Converts raw phi/theta angles
from the pipeline into binary gaze classification (looking at TV or not)
using spatial grid thresholds calibrated per TV setup.
"""

import math
from pathlib import Path

import numpy as np

from .types import TVConfig

# Grid parameters (fixed — match the .npy file structure)
H, W = 342, 608
pH, pW = 35, 53
xH, xW = 2, 2

_x, _y = np.meshgrid(np.arange(0, W, pW), np.arange(0, H, pH))
_y = _y.reshape(-1)
_x = _x.reshape(-1)


def correct_rotation(phi_theta_angle: np.ndarray) -> np.ndarray:
    """Apply head rotation correction to gaze angles.

    Args:
        phi_theta_angle: (N, 3) array of [phi, theta, head_angle_degrees]

    Returns:
        (N, 3) array of [corrected_phi, corrected_theta, head_angle]
    """
    s0 = phi_theta_angle[:, 0]
    s1 = phi_theta_angle[:, 1]
    tc_angle = phi_theta_angle[:, 2]

    x = np.cos(s1) * np.sin(s0)
    y = np.sin(s1)
    z = -np.cos(s1) * np.cos(s0)

    tangle = -(tc_angle / 180) * np.pi
    xt = np.cos(tangle) * x + np.sin(tangle) * y
    yt = -np.sin(tangle) * x + np.cos(tangle) * y
    zt = z

    vt = np.stack((xt, yt, zt), 1)
    vtnorm = (vt * vt).sum(1)
    vt = vt / np.sqrt(vtnorm.reshape(-1, 1))

    ns0 = np.arctan2(vt[:, 0], -vt[:, 2])
    ns1 = np.arcsin(vt[:, 1])

    return np.stack((ns0, ns1, tc_angle), 1)


def load_gaze_limits(npy_path: str, tv_config: TVConfig) -> np.ndarray:
    """Load and adjust spatial gaze grid limits for a TV configuration.

    Args:
        npy_path: Path to the .npy gaze limits file
        tv_config: Physical TV/camera setup parameters

    Returns:
        (N, 4) array of [phi_min, phi_max, theta_min, theta_max] per grid cell
    """
    loc_lims = np.load(npy_path).reshape(-1, 4)

    drl = (loc_lims[:, 1] - loc_lims[:, 0]) / 2.0
    dtb = (loc_lims[:, 3] - loc_lims[:, 2]) / 2.0

    # Determine TV size category
    tvs = "small" if tv_config.size < 37 else "big"

    # Camera position relative to TV
    cam_below_tv = tv_config.cam_height <= tv_config.tv_height + 5
    if cam_below_tv:
        tb_shift = 15 if tv_config.cam_height <= 35 else 10
    elif tv_config.cam_height >= 70:
        tb_shift = -20
    elif tv_config.cam_height >= 60:
        tb_shift = -15
    elif tv_config.cam_height >= 50:
        tb_shift = -10
    else:
        tb_shift = 0

    # Scale factors (center position assumed)
    slr = 1.1
    stb = 1.1
    if tvs == "big":
        rls_sc, tbs_sc = 0.3, 0.2
    else:
        rls_sc, tbs_sc = 0.1, 0.05

    rls = drl * rls_sc
    tbs = dtb * tbs_sc

    loc_lims[:, 0] = slr * loc_lims[:, 0] - rls
    loc_lims[:, 1] = slr * loc_lims[:, 1] + rls
    loc_lims[:, 2] = stb * loc_lims[:, 2] - tbs + tb_shift
    loc_lims[:, 3] = stb * loc_lims[:, 3] + tbs + tb_shift

    return loc_lims


def classify_gaze(gaze_data: np.ndarray, gaze_limits: np.ndarray) -> np.ndarray:
    """Classify gaze angles as looking-at-TV (1) or not (0).

    Args:
        gaze_data: (N, 6) array of [phi, theta, bbox_top, bbox_left, bbox_bottom, bbox_right]
        gaze_limits: (G, 4) grid limits from load_gaze_limits()

    Returns:
        (N,) array of 0 (no gaze) or 1 (gaze at TV)
    """
    import pandas as pd

    pred = 2 * np.ones(gaze_data.shape[0])

    for i in range(_y.shape[0]):
        leftl = max(_x[i], 0)
        rightl = min(_x[i] + pW + xW, W)
        topl = max(_y[i], 0)
        bottoml = min(_y[i] + pH + xH, H)

        # Coerce bbox columns to numeric
        gaze_data[:, 3] = pd.to_numeric(gaze_data[:, 3], errors="coerce")
        gaze_data[:, 5] = pd.to_numeric(gaze_data[:, 5], errors="coerce")

        col_mid = (gaze_data[:, 3] + gaze_data[:, 5]) / 2  # (left + right) / 2
        row_mid = (gaze_data[:, 2] + gaze_data[:, 4]) / 2  # (top + bottom) / 2

        in_col = np.logical_and(col_mid > leftl, col_mid < rightl)
        in_row = np.logical_and(row_mid > topl, row_mid < bottoml)
        mask = np.logical_and(in_col, in_row)

        lims = gaze_limits[i]
        if mask.sum() < 1 or any(math.isnan(l) for l in lims):
            continue

        phi_loc = gaze_data[mask, 0]
        theta_loc = gaze_data[mask, 1]

        # Threshold check: is gaze within this grid cell's angle limits?
        phi_ok = np.logical_and(
            phi_loc > (lims[0] / 180.0) * math.pi,
            phi_loc < (lims[1] / 180.0) * math.pi,
        )
        theta_ok = np.logical_and(
            theta_loc > (lims[2] / 180.0) * math.pi,
            theta_loc < (lims[3] / 180.0) * math.pi,
        )
        gaze_est = (np.logical_and(phi_ok, theta_ok)).astype(np.int32)
        pred[mask] = gaze_est

    return pred
