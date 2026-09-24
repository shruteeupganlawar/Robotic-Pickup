"""Validation report: what was wrong with a frame, and what we did about it."""

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    VALID = "VALID"                  # safe to use as-is
    RECOVERABLE = "RECOVERABLE"      # unsafe as-is, but can be repaired traceably
    UNRECOVERABLE = "UNRECOVERABLE"  # must be rejected


_SEVERITY = {Status.VALID: 0, Status.RECOVERABLE: 1, Status.UNRECOVERABLE: 2}


@dataclass
class FrameReport:
    status: Status = Status.VALID
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    actions: list = field(default_factory=list)   # repairs that will be / were applied
    stats: dict = field(default_factory=dict)

    def escalate(self, new_status: Status) -> None:
        """Status can only get worse while validating, never better."""
        if _SEVERITY[new_status] > _SEVERITY[self.status]:
            self.status = new_status

    @property
    def issues_detected(self) -> bool:
        """True if anything at all looked wrong (used for detection metrics)."""
        return bool(self.errors or self.warnings)

    @property
    def usable(self) -> bool:
        return self.status != Status.UNRECOVERABLE

    def summary(self) -> str:
        lines = ["=" * 44, "RGB-D FRAME VALIDATION", "=" * 44,
                 f"Status : {self.status.value}"]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        for a in self.actions:
            lines.append(f"  ACTION  : {a}")
        if self.stats:
            lines.append("Stats  : " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in self.stats.items()))
        return "\n".join(lines)
