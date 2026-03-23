#!/usr/bin/env python3
"""Set up a new family directory for validation.

Usage:
    python setup_family.py <family_id> \
        --vatic path/to/606_tc_gaze.txt \
        --reg path/to/606_reg.txt \
        --rot path/to/606_rot.txt \
        --start-time-file path/to/606_webcam.mp4_time_video_started.txt \
        [--end-frame 107000] \
        [--tv-size 32] [--cam-height 43] [--tv-height 50] [--view-dist 64]

    # Or with explicit start time instead of file:
    python setup_family.py 606 \
        --vatic path/to/606_tc_gaze.txt \
        --reg path/to/606_reg.txt \
        --rot path/to/606_rot.txt \
        --start-time "2023-09-25 12:54:08"

Copies the required files into <family_id>/ and creates family.yaml.
"""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Set up a family directory for validation")
    p.add_argument("family_id", help="Family ID (e.g. 606)")
    p.add_argument("--vatic", required=True, help="VATIC annotation file")
    p.add_argument("--reg", required=True, help="Pipeline _reg.txt file")
    p.add_argument("--rot", required=True, help="Pipeline _rot.txt file")

    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--start-time-file", help="Path to *_time_video_started.txt")
    start.add_argument("--start-time", help="Video start time as ISO string")

    p.add_argument("--end-frame", type=int, help="Last VATIC frame to use")
    p.add_argument("--tv-size", type=float, default=32.0)
    p.add_argument("--cam-height", type=float, default=43.0)
    p.add_argument("--tv-height", type=float, default=50.0)
    p.add_argument("--view-dist", type=float, default=64.0)

    args = p.parse_args()

    base = Path(__file__).parent
    family_dir = base / args.family_id
    family_dir.mkdir(exist_ok=True)

    # Copy files
    for src, label in [(args.vatic, "VATIC"), (args.reg, "reg"), (args.rot, "rot")]:
        dst = family_dir / Path(src).name
        if not Path(src).exists():
            print(f"ERROR: {label} file not found: {src}")
            return
        shutil.copy2(src, dst)
        print(f"  Copied {label}: {dst.name}")

    if args.start_time_file:
        dst = family_dir / Path(args.start_time_file).name
        shutil.copy2(args.start_time_file, dst)
        print(f"  Copied start time: {dst.name}")
    else:
        # Write a synthetic start time file
        dst = family_dir / f"{args.family_id}_time_video_started.txt"
        # Convert ISO to the format the parser expects
        from datetime import datetime
        dt = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
        dst.write_text(dt.strftime("%a %b %d %H:%M:%S CDT %Y") + "\n")
        print(f"  Created start time: {dst.name}")

    # Write family.yaml
    yaml_lines = []
    if args.end_frame:
        yaml_lines.append(f"end_frame: {args.end_frame}")
    yaml_lines.append("tv:")
    yaml_lines.append(f"  size: {args.tv_size}")
    yaml_lines.append(f"  cam_height: {args.cam_height}")
    yaml_lines.append(f"  tv_height: {args.tv_height}")
    yaml_lines.append(f"  view_dist: {args.view_dist}")

    yaml_path = family_dir / "family.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"  Created: family.yaml")

    print(f"\nReady. Run validation with:")
    print(f"  python -m validation evaluate --family-dir families/{args.family_id} "
          f"--gaze-limits families/4331_v3r50reg_reg_testlims_35_53_7_9.npy")


if __name__ == "__main__":
    main()
