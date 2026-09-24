import time
from dataclasses import dataclass, field
from typing import Any

from .depth import DepthConverter
from .pointcloud import intrinsics_from_proj
from .recovery import Recovery
from .report import FrameReport, Status
from .validator import FrameValidator


@dataclass
class SafeFrame:
    accepted: bool
    report: FrameReport
    rgb: Any = None
    depth_buffer: Any = None
    depth_metric: Any = None
    confidence: Any = None
    actions: list = field(default_factory=list)
    reason: str = ""
    latency_ms: float = 0.0


class SafePerception:
    def __init__(self, height, width, proj_matrix, rgb_channels=4,
                 max_invalid_fraction=0.20, reject_constant_depth=True,
                 roughness_warn=None, roughness_reject=None):
        self.height = height
        self.width = width
        self.converter = DepthConverter.from_proj_matrix(proj_matrix)
        self.intrinsics = intrinsics_from_proj(proj_matrix, width, height)
        self.validator = FrameValidator(
            height, width, rgb_channels=rgb_channels,
            max_invalid_fraction=max_invalid_fraction,
            reject_constant_depth=reject_constant_depth, roughness_warn=roughness_warn,
            roughness_reject=roughness_reject)
        self.recovery = Recovery(height, width, rgb_channels=rgb_channels)

    def process(self, rgb, depth) -> SafeFrame:
        t0 = time.perf_counter()
        report = self.validator.validate(rgb, depth)

        if report.status == Status.UNRECOVERABLE:
            return SafeFrame(False, report, reason="; ".join(report.errors),
                             latency_ms=(time.perf_counter() - t0) * 1000)

        result = self.recovery.repair(rgb, depth, report)
        if not result.accepted:
            return SafeFrame(False, report, reason=result.reason,
                             latency_ms=(time.perf_counter() - t0) * 1000)

        metric = self.converter.buffer_to_metric(result.depth)
        return SafeFrame(True, report, result.rgb, result.depth, metric,
                         result.confidence, result.actions,
                         latency_ms=(time.perf_counter() - t0) * 1000)
