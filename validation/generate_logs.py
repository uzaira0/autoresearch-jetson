#!/usr/bin/env python3
"""Generate pipeline logs (_reg.txt, _rot.txt) by running FLASH-TV on pre-extracted frames.

Edit the constants in the __main__ section at the bottom, then run:
    python generate_logs.py

Requirements:
    - Must run on the Jetson (or any machine with CUDA + the FLASH-TV models installed)
    - flash-tv-scripts repo checked out to stable-v3.8 or later

Expected directory structures
==============================

FLASH_TV_DIR — the flash-tv-scripts repo (stable-v3.8 branch):
    flash-tv-scripts/
        python_scripts/            <-- point FLASH_TV_DIR here
            flash_main.py
            flash/
                face_detection.py
                face_processing.py
                face_verification.py
                gaze_estimation.py
                net.py
            utils/
                flash_runtime_utils.py
                rotate_frame.py
                ...

    Models must be installed at their standard paths on the Jetson:
        /home/<USERNAME>/insightface/detection/RetinaFace/
        /home/<USERNAME>/Desktop/FLASH_TV_v3/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
        /home/<USERNAME>/gaze_models/model_v3_best_Gaze360ETHXrtGene_r50.pth.tar
        /home/<USERNAME>/gaze_models/model_v3_best_Gaze360ETHXrtGene_r50reg.pth.tar

DATA_DIR — root directory of family data to process:
    <DATA_DIR>/
        606/
            606_frames/                     # 1920x1080 PNGs named 000001.png, 000002.png, ...
            606_flash_log_2023-09-25.txt    # original pipeline log from data collection
            606_faces/                      # gallery reference faces
                606_tc1.png                 #   target child (up to 5)
                606_tc2.png
                ...
                606_sib1.png                #   sibling (up to 5)
                ...
                606_parent1.png             #   parent (up to 5)
                ...
                606_extra1.png              #   extra/poster (up to 5)
                ...
        401/
            401_frames/
            401_flash_log_*.txt
            401_faces/
                ...

    Auto-discovery rules:
        Frames dir:    <famid>_frames/  or  frames/
        Original log:  <famid>_flash_log*.txt  or  txts/<famid>.txt  (excludes _reg.txt, _rot.txt)
        Gallery dir:   <famid>_faces/  or  faces/

OUTPUT_DIR — where generated logs are written:
    <OUTPUT_DIR>/
        606/
            606_reg.txt                     # <-- generated
            606_rot.txt                     # <-- generated
            606_flash_log_<timestamp>.txt   # <-- generated (main log)
            606_faces/                      # symlinked from DATA_DIR
        401/
            ...

Order of operations
===================

1. EXTRACT FRAMES (if you only have video files):
       ffmpeg -i 606_webcam.mp4 -vf scale=1920:1080 -f image2 606_frames/%06d.png

2. ORGANIZE DATA_DIR so each family has frames, original log, and gallery faces.
   (See DATA_DIR structure above.)

3. EDIT THE CONSTANTS below (FLASH_TV_DIR, USERNAME, DATA_DIR, OUTPUT_DIR).

4. RUN THIS SCRIPT on the Jetson:
       python generate_logs.py
   Produces _reg.txt and _rot.txt in OUTPUT_DIR/<famid>/.

5. COPY VATIC + START TIME FILES into each OUTPUT_DIR/<famid>/:
       cp 606_tc_gaze_epoch_5s_30fps.txt      OUTPUT_DIR/606/
       cp 606_webcam.mp4_time_video_started.txt OUTPUT_DIR/606/
   Optionally create family.yaml with end_frame and tv config.

6. COPY THE SHARED GAZE LIMITS FILE into OUTPUT_DIR/:
       cp 4331_v3r50reg_reg_testlims_35_53_7_9.npy OUTPUT_DIR/

7. RUN VALIDATION:
       python -m validation batch --families-dir OUTPUT_DIR \\
           --gaze-limits OUTPUT_DIR/4331_v3r50reg_reg_testlims_35_53_7_9.npy
"""

import glob
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def find_original_log(family_dir: str, famid: str) -> str:
    """Find the original flash_log in a family directory."""
    patterns = [
        os.path.join(family_dir, f"{famid}_flash_log*.txt"),
        os.path.join(family_dir, "txts", f"{famid}.txt"),
        os.path.join(family_dir, f"{famid}.txt"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        # Exclude _reg.txt and _rot.txt
        matches = [m for m in matches if not m.endswith("_reg.txt") and not m.endswith("_rot.txt")]
        if matches:
            return matches[0]
    return ""


def find_frames_dir(family_dir: str, famid: str) -> str:
    """Find the frames directory in a family directory."""
    candidates = [
        os.path.join(family_dir, f"{famid}_frames"),
        os.path.join(family_dir, "frames"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def find_gallery_dir(family_dir: str, famid: str) -> str:
    """Find the gallery faces directory."""
    candidates = [
        os.path.join(family_dir, f"{famid}_faces"),
        os.path.join(family_dir, "faces"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


def run_on_frames(
    flash_tv_dir: str,
    famid: str,
    frames_dir: str,
    original_log: str,
    gallery_dir: str,
    output_dir: str,
    username: str = "",
):
    """Run the FLASH-TV pipeline on pre-extracted frames for one family.

    Instead of shelling out to run_flash_on_frames.py (which has hardcoded paths),
    this function sets up the environment and runs the pipeline directly.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve username
    if not username:
        username = os.environ.get("SUDO_USER", os.getlogin())

    # Symlink or copy gallery faces into output_dir so FLASHtv can find them
    gallery_link = os.path.join(output_dir, f"{famid}_faces")
    if not os.path.exists(gallery_link):
        # Use symlink if possible, copy otherwise
        try:
            os.symlink(os.path.abspath(gallery_dir), gallery_link)
        except OSError:
            shutil.copytree(gallery_dir, gallery_link)

    # Add flash-tv-scripts to path
    if flash_tv_dir not in sys.path:
        sys.path.insert(0, flash_tv_dir)

    # Import FLASH modules
    from flash_main import FLASHtv
    from utils.flash_runtime_utils import correct_rotation, write_log_file
    from utils.rotate_frame import rotate_frame

    # Read original log for frame numbers and timestamps
    with open(original_log, "r") as f:
        log_lines_input = f.readlines()
    log_lines_input = [line.strip() for line in log_lines_input if line.strip()]
    # Skip header if present
    if log_lines_input and log_lines_input[0].startswith("date"):
        log_lines_input = log_lines_input[1:]

    print(f"[{famid}] {len(log_lines_input)} frames to process from {original_log}")
    print(f"[{famid}] Frames dir: {frames_dir}")
    print(f"[{famid}] Gallery: {gallery_dir}")

    # Initialize pipeline
    flash_tv = FLASHtv(
        username,
        family_id=famid,
        num_identities=4,
        data_path=output_dir,
        frame_res_hw=None,
        output_res_hw=None,
    )
    rot_frame = rotate_frame()

    # Output log paths
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(output_dir, f"{famid}_flash_log_{timestamp_str}.txt")
    log_path_reg = os.path.join(output_dir, f"{famid}_reg.txt")
    log_path_rot = os.path.join(output_dir, f"{famid}_rot.txt")

    # Write headers
    header = "date TimeStamp frameNum numFaces tcPresent phi theta sigma rot. top left bottom right tag\n"
    for p in [log_path, log_path_reg, log_path_rot]:
        with open(p, "w") as f:
            f.write(header)

    import cv2
    import numpy as np

    batch_count = 0
    out_lines = []
    out_lines_rot = []
    out_lines_reg = []
    skipped = 0

    for line_idx, data_line in enumerate(log_lines_input):
        parts = data_line.split()
        if len(parts) < 3:
            continue

        datetime_str = parts[0] + " " + parts[1]
        try:
            time_stamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            continue
        frame_num = int(parts[2])

        # Read frame pair
        img1_path = os.path.join(frames_dir, f"{frame_num:06d}.png")
        img2_path = os.path.join(frames_dir, f"{frame_num + 1:06d}.png")

        if not os.path.exists(img1_path):
            skipped += 1
            continue

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path) if os.path.exists(img2_path) else img1.copy()

        if img1 is None:
            skipped += 1
            continue

        # Convert to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        frame_1080p_ls = [img1_rgb, img2_rgb]

        # Gallery update check
        for idx in range(4):
            time_diff = datetime.now() - flash_tv.fv.gal_updated_time[idx]
            if time_diff.total_seconds() >= 150.0:
                flash_tv.fv.gal_update[idx] = True

        # Detection
        img1_bgr, img2_bgr = img1_rgb[:, :, ::-1], img2_rgb[:, :, ::-1]
        frame_bbox_ls = [flash_tv.run_detector(img1_bgr)]
        img2_rot = rot_frame.rotate(img2_rgb)
        bbox2 = flash_tv.run_detector(img2_rot[:, :, ::-1], now_threshold=0.11)
        bbox2 = rot_frame.rotate_transform(bbox2) if rot_frame.rotate_flip >= 0 else bbox2
        frame_bbox_ls.append(bbox2)

        if any(frame_bbox_ls):
            frame_bbox_ls = [
                flash_tv.run_verification(img[:, :, ::-1], bbox_ls)
                for img, bbox_ls in zip(frame_1080p_ls, frame_bbox_ls)
            ]

            tc_present, gz_data, tc_bboxs, tc_id, tc_imgs = flash_tv.run_gaze(
                frame_1080p_ls, frame_bbox_ls
            )
            num_faces = 0 if tc_id < 0 else len(frame_bbox_ls[tc_id])

            if tc_present:
                tc_bbox = tc_bboxs[0]
                label = "Gaze-det"
                o1, e1, o2, e2 = gz_data
                gaze_data1 = list(o1[0]) + [e1[0][0]]
                gaze_data2 = list(o2[0]) + [e2[0][0]]
                tc_angle = tc_bbox["angle"]
                gaze_data1_rot = (
                    correct_rotation(gaze_data1, tc_angle)
                    if abs(tc_angle) >= 30
                    else gaze_data1
                )
                tc_pos = [tc_bbox["top"], tc_bbox["left"], tc_bbox["bottom"], tc_bbox["right"]]
            else:
                label = "Gaze-no-det"
                gaze_data1 = [None] * 3
                gaze_data2 = [None] * 3
                gaze_data1_rot = [None] * 3
                tc_pos = [None] * 4
                tc_angle = None

            write_line = (
                [time_stamp, str(frame_num).zfill(6), num_faces, int(tc_present)]
                + gaze_data1 + [tc_angle] + tc_pos + [label]
            )
            write_line_rot = (
                [time_stamp, str(frame_num).zfill(6), num_faces, int(tc_present)]
                + gaze_data1_rot + [tc_angle] + tc_pos + [label]
            )
            write_line_reg = (
                [time_stamp, str(frame_num).zfill(6), num_faces, int(tc_present)]
                + gaze_data2 + [tc_angle] + tc_pos + [label]
            )

            out_lines_rot.append(write_line_rot)
            out_lines_reg.append(write_line_reg)
        else:
            label = "No-face-detected"
            write_line = [
                time_stamp, str(frame_num).zfill(6), 0, 0,
                None, None, None, None, None, None, None, None, label,
            ]

        # Update rotation state
        num_faces_frame2 = len(frame_bbox_ls[1])
        tc_present_frame2 = tc_id == 2 if "tc_id" in dir() else False
        rot_frame.update(tc_present_frame2, num_faces_frame2)

        out_lines.append(write_line)

        # Flush every 5 lines
        if len(out_lines) >= 5:
            write_log_file(log_path, out_lines)
            write_log_file(log_path_rot, out_lines_rot)
            write_log_file(log_path_reg, out_lines_reg)
            out_lines = []
            out_lines_rot = []
            out_lines_reg = []

        batch_count += 1
        if batch_count % 100 == 0:
            print(f"[{famid}] Processed {batch_count}/{len(log_lines_input)} frames")

    # Flush remaining
    if out_lines:
        write_log_file(log_path, out_lines)
        write_log_file(log_path_rot, out_lines_rot)
        write_log_file(log_path_reg, out_lines_reg)

    print(f"[{famid}] Done. {batch_count} processed, {skipped} skipped.")
    print(f"[{famid}] Logs written to:")
    print(f"  {log_path}")
    print(f"  {log_path_reg}")
    print(f"  {log_path_rot}")

    return log_path_reg, log_path_rot


if __name__ == "__main__":

    # ── EDIT THESE ────────────────────────────────────────────────────
    #
    # Path to flash-tv-scripts/python_scripts (the repo with flash_main.py)
    FLASH_TV_DIR = "/home/flashsys/flash-tv-scripts/python_scripts"
    #
    # Username for model paths (/home/<USERNAME>/gaze_models, etc.)
    # Leave empty to auto-detect.
    USERNAME = ""
    #
    # Root directory containing family subdirectories.
    # Each subdir must contain:
    #   <famid>_frames/          PNGs at 1920x1080
    #   <famid>_flash_log*.txt   original log (frame numbers + timestamps)
    #   <famid>_faces/           gallery faces (tc1-5, sib1-5, parent1-5, extra1-5)
    DATA_DIR = "/path/to/data"
    #
    # Where to write the output _reg.txt / _rot.txt logs.
    # Each family gets a subdir: OUTPUT_DIR/<famid>/
    OUTPUT_DIR = "/path/to/families"
    #
    # ──────────────────────────────────────────────────────────────────

    batch_dir = Path(DATA_DIR)
    families = sorted([
        d.name for d in batch_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(families)} family directories in {batch_dir}")

    for famid in families:
        fam_dir = str(batch_dir / famid)

        frames_dir = find_frames_dir(fam_dir, famid)
        original_log = find_original_log(fam_dir, famid)
        gallery_dir = find_gallery_dir(fam_dir, famid)

        missing = []
        if not frames_dir:
            missing.append(f"{famid}_frames/")
        if not original_log:
            missing.append(f"{famid}_flash_log*.txt")
        if not gallery_dir:
            missing.append(f"{famid}_faces/")

        if missing:
            print(f"SKIP {famid}: missing {', '.join(missing)}")
            continue

        out_dir = os.path.join(OUTPUT_DIR, famid)
        print(f"\n{'='*60}")
        print(f"Processing family {famid}")
        print(f"{'='*60}")

        try:
            run_on_frames(
                flash_tv_dir=FLASH_TV_DIR,
                famid=famid,
                frames_dir=frames_dir,
                original_log=original_log,
                gallery_dir=gallery_dir,
                output_dir=out_dir,
                username=USERNAME,
            )
        except Exception as e:
            print(f"ERROR {famid}: {e}")
            import traceback
            traceback.print_exc()
