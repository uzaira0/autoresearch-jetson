"""Parsers for VATIC annotations, pipeline logs, and discovery helpers."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import BBox, PipelineFrame, VATICAnnotation


# ── VATIC parsing ─────────────────────────────────────────────────────

def parse_vatic(filepath: str) -> List[VATICAnnotation]:
    """Parse a VATIC annotation dump file.

    Format: track_id xmin ymin xmax ymax frame lost occluded generated "label" ["attr"...]
    """
    annotations = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"VATIC file not found: {filepath}")

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            quoted = re.findall(r'"([^"]*)"', line)
            if not quoted:
                raise ValueError(f"{filepath}:{line_num}: No quoted label found")
            numeric_part = line[: line.index('"')].strip()
            nums = numeric_part.split()
            if len(nums) != 9:
                raise ValueError(f"{filepath}:{line_num}: Expected 9 numeric fields, got {len(nums)}")
            annotations.append(VATICAnnotation(
                track_id=int(nums[0]),
                bbox=BBox(int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4])),
                frame=int(nums[5]),
                lost=bool(int(nums[6])),
                occluded=bool(int(nums[7])),
                generated=bool(int(nums[8])),
                label=quoted[0],
                attributes=quoted[1:],
            ))
    return annotations


def vatic_to_timeseries(
    filepath: str,
    video_start_time: str,
    fps: int = 30,
    end_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Convert VATIC annotations to a 1-per-second gaze time series.

    This is the core transformation: frame numbers → timestamps → resampled.

    Returns DataFrame indexed by timestamp with column 'gaze' (1=Gaze, 0=No-Gaze).
    """
    # Read only frame number (col 5) and gaze attribute (col 10)
    data = pd.read_csv(
        filepath, sep=" ", header=None,
        usecols=[5, 10], names=["frame", "category"],
    )
    data["category"] = data["category"].map({
        '"Gaze"': 1, '"No-Gaze"': 0, '"Uncertain"': -1, '"Out-of-Frame"': 0,
        "Gaze": 1, "No-Gaze": 0, "Uncertain": -1, "Out-of-Frame": 0,
    })

    if end_frame is not None:
        data = data[data["frame"] <= end_frame]

    # Convert frame number to timestamp
    start = pd.to_datetime(video_start_time)
    data["timestamp"] = start + pd.to_timedelta(data["frame"] // fps, unit="s")

    # Deduplicate (multiple tracks per frame → keep first gaze opinion per second)
    data = data.drop_duplicates(subset="timestamp")
    data = data.set_index("timestamp")

    # Resample to 1/sec via forward-fill
    full_range = pd.date_range(start=start, end=data.index.max(), freq="s")
    result = data.reindex(full_range, method="ffill").reset_index()
    result.columns = ["timestamp", "frame", "gaze"]

    return result.set_index("timestamp")[["gaze"]]


# ── Pipeline log parsing ──────────────────────────────────────────────

PIPELINE_COLUMNS = [
    "date", "TimeStamp", "frameNum", "numFaces", "tcPresent",
    "phi", "theta", "sigma", "rot.", "top", "left", "bottom", "right", "tag",
]


def parse_pipeline_log(filepath: str) -> pd.DataFrame:
    """Parse a pipeline log file (_reg.txt or _rot.txt) into a DataFrame.

    Handles both 13-field (no bbox) and 14-field (with bbox) lines.
    Handles files with and without a header row.
    Returns DataFrame indexed by timestamp (floored to seconds).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline log not found: {filepath}")

    # Check for header
    with open(path) as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith("date")

    df = pd.read_csv(
        filepath, sep=r"\s+",
        header=0 if has_header else None,
        names=None if has_header else PIPELINE_COLUMNS,
    )

    # Handle short lines where tag landed in 'right' column
    condition = df["tag"].isna()
    if condition.any():
        df.loc[condition, "tag"] = df.loc[condition, "right"]
        df.loc[condition, "right"] = float("nan")

    df["right"] = pd.to_numeric(df["right"], errors="coerce")
    df["tag"] = df["tag"].astype(str)

    # Build timestamp index
    df["dateTimeStamp"] = df["date"] + " " + df["TimeStamp"]
    df["dateTimeStamp"] = pd.to_datetime(df["dateTimeStamp"])
    df = df.drop(columns=["date", "TimeStamp"])
    df = df.set_index("dateTimeStamp").sort_index()
    df.index = df.index.floor("s")

    return df


def parse_flash_log_raw(filepath: str) -> List[PipelineFrame]:
    """Parse a flash_log file into a list of PipelineFrame objects.

    Used by the `diff` command. For epoch-based evaluation, use
    parse_pipeline_log() instead.
    """
    frames = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Flash log not found: {filepath}")

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("date"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue

            def _opt_float(v):
                return None if v.lower() == "none" else float(v)

            def _opt_int(v):
                return None if v.lower() == "none" else int(float(v))

            timestamp = f"{parts[0]} {parts[1]}"
            tc_bbox = None
            if len(parts) >= 14:
                t, l, b, r = _opt_int(parts[9]), _opt_int(parts[10]), _opt_int(parts[11]), _opt_int(parts[12])
                if all(v is not None for v in [t, l, b, r]):
                    tc_bbox = BBox(xmin=l, ymin=t, xmax=r, ymax=b)
                status = parts[13]
            else:
                status = parts[12]

            frames.append(PipelineFrame(
                timestamp=timestamp,
                frame_id=int(parts[2]),
                num_faces=int(parts[3]),
                tc_identified=int(parts[4]) == 1,
                gaze_phi=_opt_float(parts[5]),
                gaze_theta=_opt_float(parts[6]),
                gaze_error=_opt_float(parts[7]),
                head_angle=_opt_float(parts[8]),
                tc_bbox=tc_bbox,
                status=status,
            ))
    return frames


# ── Video start time parsing ──────────────────────────────────────────

def parse_video_start_time(filepath: str) -> str:
    """Parse a *_time_video_started.txt file.

    These contain a single line like: 'Mon Sep 25 12:54:08 CDT 2023'
    Returns ISO format string: '2023-09-25 12:54:08'
    """
    with open(filepath) as f:
        raw = f.read().strip()

    # Remove timezone abbreviation (CDT, CST, EST, etc.) — not recognized by strptime
    raw_no_tz = re.sub(r"\b[A-Z]{3,4}\b", "", raw).strip()
    # Collapse multiple spaces
    raw_no_tz = re.sub(r"\s+", " ", raw_no_tz)

    dt = datetime.strptime(raw_no_tz, "%a %b %d %H:%M:%S %Y")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Discovery ─────────────────────────────────────────────────────────

def discover_family_files(family_dir: str) -> Dict[str, Optional[str]]:
    """Auto-discover files in a family directory by naming convention.

    Returns dict with keys: 'vatic', 'reg', 'rot', 'start_time', 'main_log'
    Values are file paths or None if not found.
    """
    d = Path(family_dir)
    result = {"vatic": None, "reg": None, "rot": None, "start_time": None, "main_log": None}

    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        name = f.name.lower()

        if name.endswith("_time_video_started.txt"):
            result["start_time"] = str(f)
        elif name.endswith("_reg.txt") and "flash" not in name:
            result["reg"] = str(f)
        elif name.endswith("_rot.txt"):
            result["rot"] = str(f)
        elif any(p in name for p in ["tcgz", "tc_gz", "tc_gaze"]) and name.endswith(".txt"):
            # Prefer the most specific annotation file
            if result["vatic"] is None or "epoch" in name:
                result["vatic"] = str(f)
        elif re.match(r"^\d+\.txt$", name):
            # Main log: just digits.txt (e.g., 606.txt)
            result["main_log"] = str(f)

    return result


def extract_family_id(filepath: str) -> str:
    """Extract family ID from filename. '606_tc_gaze.txt' → '606'."""
    match = re.match(r"^(\d+)", Path(filepath).stem)
    return match.group(1) if match else Path(filepath).stem
