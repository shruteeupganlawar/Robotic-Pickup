"""Recovery: repair what can be repaired safely, and record every change.

Principles: never modify silently, and prefer rejection to invented data.
Repairs never touch pixels that were valid. Every repaired pixel gets a LOW confidence
that decays with its distance to the nearest valid pixel.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .report import FrameReport, Status
from .reshaper import SafeReshaper
from .validator import depth_invalid_mask


@dataclass
class RecoveryResult:
    accepted: bool
    rgb: Any = None
    depth: Any = None            # repaired depth, still in buffer space
    confidence: Any = None       # (H, W) in [0, 1]
    actions: list = field(default_factory=list)
    reason: str = ""


class Recovery:
    def __init__(self, height, width, rgb_channels=4, inpaint_radius=3,
                 repaired_confidence=0.6, confidence_decay_px=6.0):
        self.reshaper = SafeReshaper(height, width, rgb_channels)
        self.inpaint_radius = inpaint_radius
        self.repaired_confidence = repaired_confidence
        self.decay = confidence_decay_px

    def repair(self, rgb, depth, report: FrameReport) -> RecoveryResult:
        if report.status == Status.UNRECOVERABLE:
            return RecoveryResult(False, reason="; ".join(report.errors))

        actions = []
        rgb_r = self.reshaper.reshape_rgb(rgb)
        if "reshape_rgb" in report.actions:
            actions.append("reshape_rgb")
        depth_r = np.array(self.reshaper.reshape_depth(depth), dtype=np.float32)  # copy
        if "reshape_depth" in report.actions:
            actions.append("reshape_depth")

        confidence = np.ones(depth_r.shape, dtype=np.float32)
        invalid = depth_invalid_mask(depth_r)

        if invalid.any():
            valid_vals = depth_r[~invalid]
            work = depth_r.copy()
            work[invalid] = np.median(valid_vals)
            mask8 = invalid.astype(np.uint8) * 255
            filled = cv2.inpaint(work, mask8, self.inpaint_radius, cv2.INPAINT_NS)
            depth_r[invalid] = np.clip(filled[invalid], valid_vals.min(), valid_vals.max())

            dist = cv2.distanceTransform(mask8, cv2.DIST_L2, 3)  # px to nearest valid pixel
            confidence[invalid] = self.repaired_confidence * np.exp(
                -np.maximum(dist[invalid] - 1.0, 0.0) / self.decay)
            actions.append(f"inpaint_depth({int(invalid.sum())} px)")

        if "median_filter" in report.actions:
            depth_r = cv2.medianBlur(depth_r, 3)
            confidence = np.minimum(confidence, 0.9)
            actions.append("median_filter(3x3)")

        return RecoveryResult(True, rgb_r, depth_r, confidence, actions)
