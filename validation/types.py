"""Data types for the FLASH-TV validation harness."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Bounding box ──────────────────────────────────────────────────────

@dataclass
class BBox:
    """Bounding box in pixel coordinates."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def scale(self, sx: float, sy: float) -> "BBox":
        return BBox(int(self.xmin * sx), int(self.ymin * sy),
                    int(self.xmax * sx), int(self.ymax * sy))

    @property
    def area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def iou(self, other: "BBox") -> float:
        xi1 = max(self.xmin, other.xmin)
        yi1 = max(self.ymin, other.ymin)
        xi2 = min(self.xmax, other.xmax)
        yi2 = min(self.ymax, other.ymax)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = self.area + other.area - inter
        return inter / union if union else 0.0


# ── VATIC annotation ──────────────────────────────────────────────────

@dataclass
class VATICAnnotation:
    """Single VATIC annotation row (one track, one frame)."""
    track_id: int
    bbox: BBox
    frame: int
    lost: bool
    occluded: bool
    generated: bool
    label: str
    attributes: List[str] = field(default_factory=list)

    @property
    def gaze(self) -> Optional[bool]:
        attrs = [a.lower() for a in self.attributes]
        if "gaze" in attrs:
            return True
        if "no-gaze" in attrs:
            return False
        return None

    @property
    def out_of_frame(self) -> bool:
        return any(a.lower() == "out-of-frame" for a in self.attributes)

    @property
    def uncertain(self) -> bool:
        return any(a.lower() == "uncertain" for a in self.attributes)


# ── Pipeline frame ────────────────────────────────────────────────────

@dataclass
class PipelineFrame:
    """Single row from a FLASH-TV flash_log file."""
    timestamp: str
    frame_id: int
    num_faces: int
    tc_identified: bool
    gaze_phi: Optional[float]
    gaze_theta: Optional[float]
    gaze_error: Optional[float]
    head_angle: Optional[float]
    tc_bbox: Optional[BBox]
    status: str


# ── Family input contract ─────────────────────────────────────────────

@dataclass
class TVConfig:
    """Physical TV/camera setup for gaze threshold computation."""
    size: float = 32.0          # TV diagonal in inches
    cam_height: float = 43.0    # camera height in inches from floor
    tv_height: float = 50.0     # TV center height in inches from floor
    view_dist: float = 64.0     # viewing distance in inches


@dataclass
class FamilyInput:
    """Everything needed to evaluate one family.

    This is the input contract. Provide these fields and the harness
    handles the rest.
    """
    family_id: str
    vatic_file: str                     # path to *_tcgz.txt or *_tc_gaze_*.txt
    pipeline_reg_file: str              # path to *_reg.txt
    pipeline_rot_file: str              # path to *_rot.txt
    video_start_time: str               # ISO format: "2023-09-25 12:54:08"
    tv_config: TVConfig = field(default_factory=TVConfig)
    gaze_limits_file: str = ""          # path to .npy gaze grid limits
    fps: int = 30                       # video frame rate
    end_frame: Optional[int] = None     # optional: only use VATIC frames up to this


# ── Evaluation results ────────────────────────────────────────────────

@dataclass
class EpochResult:
    """Evaluation metrics from epoch-based comparison."""
    family_id: str
    total_epochs: int               # total 5-sec epochs compared
    gaze_epochs: int                # epochs where both have gaze opinion

    # Confusion matrix values (pipeline prediction vs VATIC ground truth)
    true_pos: int                   # both say gaze
    true_neg: int                   # both say no-gaze
    false_pos: int                  # pipeline says gaze, VATIC says no
    false_neg: int                  # pipeline says no-gaze, VATIC says gaze

    sensitivity: float              # TP / (TP + FN) — recall of gaze
    specificity: float              # TN / (TN + FP) — recall of no-gaze
    accuracy: float                 # (TP + TN) / total

    pipeline_gaze_seconds: float    # total gaze time from pipeline
    vatic_gaze_seconds: float       # total gaze time from VATIC

    def summary(self) -> str:
        lines = [
            f"=== {self.family_id} ===",
            f"Epochs compared: {self.total_epochs}  ({self.total_epochs * 5}s)",
            f"",
            f"Confusion matrix (Pipeline vs VATIC ground truth):",
            f"                    VATIC=NoGaze  VATIC=Gaze",
            f"  Pipeline=NoGaze     {self.true_neg:>6}        {self.false_neg:>6}",
            f"  Pipeline=Gaze       {self.false_pos:>6}        {self.true_pos:>6}",
            f"",
            f"Sensitivity (recall): {self.sensitivity:.3f}",
            f"Specificity:          {self.specificity:.3f}",
            f"Accuracy:             {self.accuracy:.3f}",
            f"",
            f"Gaze time (pipeline): {self.pipeline_gaze_seconds:.0f}s",
            f"Gaze time (VATIC):    {self.vatic_gaze_seconds:.0f}s",
        ]
        return "\n".join(lines)


@dataclass
class BatchResult:
    """Aggregate results across multiple families."""
    families: List[EpochResult]

    @property
    def total_epochs(self) -> int:
        return sum(f.total_epochs for f in self.families)

    @property
    def overall_accuracy(self) -> float:
        tp = sum(f.true_pos for f in self.families)
        tn = sum(f.true_neg for f in self.families)
        fp = sum(f.false_pos for f in self.families)
        fn = sum(f.false_neg for f in self.families)
        total = tp + tn + fp + fn
        return (tp + tn) / total if total else 0.0

    @property
    def overall_sensitivity(self) -> float:
        tp = sum(f.true_pos for f in self.families)
        fn = sum(f.false_neg for f in self.families)
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def overall_specificity(self) -> float:
        tn = sum(f.true_neg for f in self.families)
        fp = sum(f.false_pos for f in self.families)
        return tn / (tn + fp) if (tn + fp) else 0.0

    def summary(self) -> str:
        lines = []
        for f in self.families:
            lines.append(f.summary())
            lines.append("")

        lines.append("=== AGGREGATE ===")
        lines.append(f"Families: {len(self.families)}")
        lines.append(f"Total epochs: {self.total_epochs}  ({self.total_epochs * 5}s)")
        lines.append(f"Overall sensitivity: {self.overall_sensitivity:.3f}")
        lines.append(f"Overall specificity: {self.overall_specificity:.3f}")
        lines.append(f"Overall accuracy:    {self.overall_accuracy:.3f}")

        total_pl = sum(f.pipeline_gaze_seconds for f in self.families)
        total_gt = sum(f.vatic_gaze_seconds for f in self.families)
        lines.append(f"Total gaze (pipeline): {total_pl:.0f}s")
        lines.append(f"Total gaze (VATIC):    {total_gt:.0f}s")
        return "\n".join(lines)


# ── Pipeline diff (two runs, no ground truth) ─────────────────────────

@dataclass
class DiffResult:
    """Result of comparing two pipeline runs against each other."""
    total_frames: int
    matched_frames: int
    tc_id_agree: int
    tc_id_disagree: int
    detection_count_differ: int
    gaze_agree: int
    gaze_disagree: int
    mean_bbox_iou: Optional[float]
    bbox_frames: int

    def summary(self) -> str:
        lines = [
            f"=== Pipeline Diff ===",
            f"Frames compared: {self.matched_frames}/{self.total_frames}",
            f"TC ID agreement:      {self.tc_id_agree}/{self.matched_frames}",
            f"Detection count same: {self.matched_frames - self.detection_count_differ}/{self.matched_frames}",
        ]
        gaze_total = self.gaze_agree + self.gaze_disagree
        if gaze_total > 0:
            lines.append(f"Gaze agreement:       {self.gaze_agree}/{gaze_total}")
        if self.mean_bbox_iou is not None:
            lines.append(f"Mean TC bbox IoU:     {self.mean_bbox_iou:.3f}  ({self.bbox_frames} frames)")
        return "\n".join(lines)
