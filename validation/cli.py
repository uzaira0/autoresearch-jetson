#!/usr/bin/env python3
"""CLI for FLASH-TV validation harness.

Usage:

    # Evaluate a single family
    python -m validation evaluate \\
        --family-dir path/to/606/ \\
        --gaze-limits path/to/4331_*.npy

    # Batch evaluate all families in a directory
    python -m validation batch \\
        --families-dir path/to/families/ \\
        --gaze-limits path/to/4331_*.npy

    # Batch evaluate from a YAML config
    python -m validation batch --config config.yaml

    # Compare two pipeline runs (no ground truth)
    python -m validation diff \\
        --baseline baseline_log.txt \\
        --optimized optimized_log.txt

    # Inspect VATIC annotations
    python -m validation inspect --vatic 401_tcgz.txt --frames 0-100
"""

import argparse
import json
import sys
from pathlib import Path

from . import compare, evaluate, parsers
from .types import BatchResult, EpochResult, FamilyInput, TVConfig


# ── evaluate ──────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate one family from a directory."""
    family = _family_from_dir(
        args.family_dir,
        gaze_limits=args.gaze_limits,
        tv_config=_parse_tv_config(args),
    )
    result = evaluate.evaluate_family(family)
    _print_result(result, args)


# ── batch ─────────────────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> None:
    """Batch evaluate multiple families."""
    if args.config:
        families = _families_from_config(args.config, args)
    elif args.families_dir:
        families = _families_from_dir(args.families_dir, args)
    else:
        print("ERROR: provide --config or --families-dir")
        sys.exit(1)

    results = []
    for family in families:
        try:
            result = evaluate.evaluate_family(family)
            results.append(result)
        except Exception as e:
            print(f"SKIP {family.family_id}: {e}")

    if not results:
        print("No families evaluated successfully.")
        sys.exit(1)

    batch = BatchResult(families=results)
    if args.json:
        _print_batch_json(batch)
    else:
        print(batch.summary())


# ── diff ──────────────────────────────────────────────────────────────

def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two pipeline runs frame-by-frame."""
    baseline = parsers.parse_flash_log_raw(args.baseline)
    optimized = parsers.parse_flash_log_raw(args.optimized)

    bl_idx = {f.frame_id: f for f in baseline}
    opt_idx = {f.frame_id: f for f in optimized}

    result = compare.diff_pipeline_runs(bl_idx, opt_idx)

    if args.json:
        print(json.dumps({
            "total_frames": result.total_frames,
            "matched_frames": result.matched_frames,
            "tc_id_agree": result.tc_id_agree,
            "tc_id_disagree": result.tc_id_disagree,
            "detection_count_differ": result.detection_count_differ,
            "gaze_agree": result.gaze_agree,
            "gaze_disagree": result.gaze_disagree,
            "mean_bbox_iou": result.mean_bbox_iou,
            "bbox_frames": result.bbox_frames,
        }, indent=2))
    else:
        print(result.summary())

    if result.tc_id_disagree > 0 or result.detection_count_differ > 0:
        sys.exit(1)


# ── inspect ───────────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> None:
    """Dump parsed VATIC annotations for manual review."""
    annotations = parsers.parse_vatic(args.vatic)
    ann_type = _detect_type(args.vatic)
    fam_id = parsers.extract_family_id(args.vatic)

    by_frame = {}
    for a in annotations:
        by_frame.setdefault(a.frame, []).append(a)

    tracks = {a.track_id for a in annotations}
    frames = sorted(by_frame.keys())

    print(f"File:      {args.vatic}")
    print(f"Type:      {ann_type}")
    print(f"Family:    {fam_id}")
    print(f"Rows:      {len(annotations)}")
    print(f"Frames:    {min(frames)}..{max(frames)}")
    print(f"Track IDs: {sorted(tracks)}")
    print()

    if args.frames:
        lo, hi = args.frames.split("-")
        frames = [f for f in frames if int(lo) <= f <= int(hi)]

    for f in frames[: args.limit]:
        for a in by_frame[f]:
            g = " GAZE" if a.gaze is True else (" NO-GAZE" if a.gaze is False else "")
            lost = " LOST" if a.lost else ""
            gen = " gen" if a.generated else " KEY"
            print(f"  frame={f:6d} track={a.track_id} "
                  f"bbox=({a.bbox.xmin},{a.bbox.ymin},{a.bbox.xmax},{a.bbox.ymax})"
                  f"{lost}{gen}{g}")


# ── helpers ───────────────────────────────────────────────────────────

def _family_from_dir(
    family_dir: str,
    gaze_limits: str = "",
    tv_config: TVConfig = None,
) -> FamilyInput:
    """Build a FamilyInput from auto-discovered files in a directory."""
    found = parsers.discover_family_files(family_dir)

    missing = []
    if not found["vatic"]:
        missing.append("VATIC annotation (*_tcgz.txt / *_tc_gaze_*.txt)")
    if not found["reg"]:
        missing.append("Pipeline regression log (*_reg.txt)")
    if not found["rot"]:
        missing.append("Pipeline rotation log (*_rot.txt)")
    if not found["start_time"]:
        missing.append("Video start time (*_time_video_started.txt)")
    if missing:
        raise FileNotFoundError(
            f"Missing files in {family_dir}:\n  " + "\n  ".join(missing)
        )

    video_start = parsers.parse_video_start_time(found["start_time"])
    fam_id = parsers.extract_family_id(found["vatic"])

    # Check for family.yaml overrides
    family_yaml = Path(family_dir) / "family.yaml"
    end_frame = None
    if family_yaml.exists():
        import yaml
        with open(family_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        end_frame = cfg.get("end_frame")
        if "tv" in cfg and tv_config is None:
            tv_config = TVConfig(**cfg["tv"])
        if "video_start_time" in cfg:
            video_start = cfg["video_start_time"]

    return FamilyInput(
        family_id=fam_id,
        vatic_file=found["vatic"],
        pipeline_reg_file=found["reg"],
        pipeline_rot_file=found["rot"],
        video_start_time=video_start,
        tv_config=tv_config or TVConfig(),
        gaze_limits_file=gaze_limits,
        end_frame=end_frame,
    )


def _families_from_dir(families_dir: str, args: argparse.Namespace) -> list:
    """Build FamilyInput for each subdirectory in families_dir."""
    base = Path(families_dir)
    tv_config = _parse_tv_config(args)
    families = []

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        try:
            family = _family_from_dir(
                str(subdir),
                gaze_limits=args.gaze_limits,
                tv_config=tv_config,
            )
            families.append(family)
        except FileNotFoundError as e:
            print(f"SKIP {subdir.name}: {e}")

    return families


def _families_from_config(config_path: str, args: argparse.Namespace) -> list:
    """Build FamilyInput list from a YAML config file.

    Config format:
        gaze_limits: path/to/file.npy

        defaults:
          fps: 30
          tv:
            size: 32.0
            cam_height: 43.0
            tv_height: 50.0
            view_dist: 64.0

        families:
          606:
            vatic: path/to/606_tc_gaze.txt
            reg: path/to/606_reg.txt
            rot: path/to/606_rot.txt
            video_start_time: "2023-09-25 12:54:08"
            # OR
            start_time_file: path/to/606_webcam.mp4_time_video_started.txt
            end_frame: 107000     # optional
            tv:                   # optional, overrides defaults
              size: 50.0
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent
    gaze_limits = args.gaze_limits or _resolve(cfg.get("gaze_limits", ""), base)
    defaults = cfg.get("defaults", {})
    default_fps = defaults.get("fps", 30)
    default_tv = TVConfig(**defaults["tv"]) if "tv" in defaults else TVConfig()

    families = []
    for fam_id, fam_cfg in cfg.get("families", {}).items():
        fam_id = str(fam_id)

        # Resolve video start time
        if "video_start_time" in fam_cfg:
            start_time = fam_cfg["video_start_time"]
        elif "start_time_file" in fam_cfg:
            start_time = parsers.parse_video_start_time(
                _resolve(fam_cfg["start_time_file"], base)
            )
        else:
            raise ValueError(f"Family {fam_id}: needs video_start_time or start_time_file")

        # TV config
        tv = TVConfig(**fam_cfg["tv"]) if "tv" in fam_cfg else default_tv

        families.append(FamilyInput(
            family_id=fam_id,
            vatic_file=_resolve(fam_cfg["vatic"], base),
            pipeline_reg_file=_resolve(fam_cfg["reg"], base),
            pipeline_rot_file=_resolve(fam_cfg["rot"], base),
            video_start_time=start_time,
            tv_config=tv,
            gaze_limits_file=gaze_limits,
            fps=fam_cfg.get("fps", default_fps),
            end_frame=fam_cfg.get("end_frame"),
        ))

    return families


def _resolve(path: str, base: Path) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(base / p)


def _parse_tv_config(args: argparse.Namespace) -> TVConfig:
    tv = TVConfig()
    if hasattr(args, "tv_size") and args.tv_size is not None:
        tv.size = args.tv_size
    if hasattr(args, "cam_height") and args.cam_height is not None:
        tv.cam_height = args.cam_height
    if hasattr(args, "tv_height") and args.tv_height is not None:
        tv.tv_height = args.tv_height
    if hasattr(args, "view_dist") and args.view_dist is not None:
        tv.view_dist = args.view_dist
    return tv


def _detect_type(filepath: str) -> str:
    name = Path(filepath).stem.lower()
    if "sib_par" in name:
        return "sib_par_gaze"
    if "tcgz" in name or "tc_gz" in name or "tc_gaze" in name:
        return "tc_gaze"
    if "tcbbx" in name or "tc_bbx" in name:
        return "tc_bbox"
    return "other"


def _print_result(result: EpochResult, args: argparse.Namespace) -> None:
    if hasattr(args, "json") and args.json:
        print(json.dumps({
            "family_id": result.family_id,
            "total_epochs": result.total_epochs,
            "true_pos": result.true_pos, "true_neg": result.true_neg,
            "false_pos": result.false_pos, "false_neg": result.false_neg,
            "sensitivity": result.sensitivity,
            "specificity": result.specificity,
            "accuracy": result.accuracy,
            "pipeline_gaze_seconds": result.pipeline_gaze_seconds,
            "vatic_gaze_seconds": result.vatic_gaze_seconds,
        }, indent=2))
    else:
        print(result.summary())


def _print_batch_json(batch: BatchResult) -> None:
    print(json.dumps({
        "families": [{
            "family_id": f.family_id,
            "total_epochs": f.total_epochs,
            "sensitivity": f.sensitivity,
            "specificity": f.specificity,
            "accuracy": f.accuracy,
        } for f in batch.families],
        "aggregate": {
            "families": len(batch.families),
            "total_epochs": batch.total_epochs,
            "overall_sensitivity": batch.overall_sensitivity,
            "overall_specificity": batch.overall_specificity,
            "overall_accuracy": batch.overall_accuracy,
        }
    }, indent=2))


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="validation", description="FLASH-TV validation harness")
    sub = p.add_subparsers(dest="command", required=True)

    # -- evaluate
    ev = sub.add_parser("evaluate", help="Evaluate one family")
    ev.add_argument("--family-dir", required=True, help="Directory containing family files")
    ev.add_argument("--gaze-limits", required=True, help="Path to .npy gaze limits file")
    ev.add_argument("--tv-size", type=float, help="TV size in inches (default 32)")
    ev.add_argument("--cam-height", type=float, help="Camera height in inches (default 43)")
    ev.add_argument("--tv-height", type=float, help="TV height in inches (default 50)")
    ev.add_argument("--view-dist", type=float, help="Viewing distance in inches (default 64)")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    # -- batch
    ba = sub.add_parser("batch", help="Batch evaluate families")
    ba.add_argument("--families-dir", help="Directory of family subdirectories")
    ba.add_argument("--config", help="YAML config file")
    ba.add_argument("--gaze-limits", default="", help="Path to .npy gaze limits file")
    ba.add_argument("--tv-size", type=float)
    ba.add_argument("--cam-height", type=float)
    ba.add_argument("--tv-height", type=float)
    ba.add_argument("--view-dist", type=float)
    ba.add_argument("--json", action="store_true")
    ba.set_defaults(func=cmd_batch)

    # -- diff
    di = sub.add_parser("diff", help="Compare two pipeline runs")
    di.add_argument("--baseline", required=True)
    di.add_argument("--optimized", required=True)
    di.add_argument("--json", action="store_true")
    di.set_defaults(func=cmd_diff)

    # -- inspect
    ins = sub.add_parser("inspect", help="Inspect VATIC annotations")
    ins.add_argument("--vatic", required=True)
    ins.add_argument("--frames", help="Frame range, e.g. '0-100'")
    ins.add_argument("--limit", type=int, default=50)
    ins.set_defaults(func=cmd_inspect)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
