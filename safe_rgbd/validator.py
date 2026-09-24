import cv2
import numpy as np

from .report import FrameReport, Status


def depth_invalid_mask(depth: np.ndarray) -> np.ndarray:
    """Pixels that cannot be trusted: NaN, Inf, <= 0, or > 1 (outside the buffer range)."""
    d = np.asarray(depth)
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(d) | (d <= 0.0) | (d > 1.0)


def depth_roughness(depth: np.ndarray) -> float:
    """Mean |d - median3x3(d)|: near zero on smooth depth, large on speckle noise.

    Invalid pixels (NaN/Inf/0/out-of-range) are ignored: only pixels whose whole 3x3
    neighbourhood is valid contribute, so holes do not masquerade as noise.
    """
    d = np.ascontiguousarray(depth, dtype=np.float32)
    invalid = depth_invalid_mask(d)
    if invalid.all():
        return float("nan")
    keep = None
    if invalid.any():
        d = d.copy()
        d[invalid] = np.median(d[~invalid])          # only so medianBlur is well-defined
        keep = cv2.erode((~invalid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    resid = np.abs(d - cv2.medianBlur(d, 3))
    if keep is not None and keep.any():
        return float(resid[keep].mean())
    return float(resid.mean())


class FrameValidator:
    def __init__(self, height, width, rgb_channels=4, max_invalid_fraction=0.20,
                 const_std_eps=1e-6, reject_constant_depth=True,
                 roughness_warn=None, roughness_reject=None):
        self.height = height
        self.width = width
        self.rgb_channels = rgb_channels
        self.max_invalid_fraction = max_invalid_fraction
        self.const_std_eps = const_std_eps
        self.reject_constant_depth = reject_constant_depth
        self.roughness_warn = roughness_warn       # None = speckle check disabled
        self.roughness_reject = roughness_reject   # None = never reject for noise alone
        # For Gaussian speckle, roughness is about 0.6 x sigma (depth-buffer units).

    def calibrate(self, clean_depth_frames, margin=1.5):
        """Set roughness_warn from known-clean frames (threshold from data, not a guess)."""
        vals = [depth_roughness(np.asarray(d).reshape(self.height, self.width))
                for d in clean_depth_frames]
        self.roughness_warn = float(max(vals) * margin)
        return self.roughness_warn

    def validate(self, rgb, depth) -> FrameReport:
        report = FrameReport()
        self._check_rgb(rgb, report)
        self._check_depth(depth, report)
        return report

    def _check_rgb(self, rgb, report):
        h, w, c = self.height, self.width, self.rgb_channels
        if rgb is None:
            report.errors.append("RGB is None")
            report.escalate(Status.UNRECOVERABLE)
            return
        a = np.asarray(rgb)
        report.stats["rgb_shape"] = tuple(a.shape)

        if a.shape == (h, w, c):
            pass
        elif a.ndim == 1 and a.size == h * w * c:
            report.errors.append(
                f"RGB is a flat buffer {a.shape}; element count matches ({h},{w},{c})")
            report.actions.append("reshape_rgb")
            report.escalate(Status.RECOVERABLE)
        else:
            report.errors.append(
                f"RGB shape {a.shape} (size {a.size}) cannot be mapped to "
                f"({h},{w},{c}): data missing or wrong format")
            report.escalate(Status.UNRECOVERABLE)
            return

        if a.dtype != np.uint8:
            report.warnings.append(f"RGB dtype is {a.dtype}, expected uint8")

    def _check_depth(self, depth, report):
        h, w = self.height, self.width
        if depth is None:
            report.errors.append("Depth is None")
            report.escalate(Status.UNRECOVERABLE)
            return
        a = np.asarray(depth)
        report.stats["depth_shape"] = tuple(a.shape)

        if not np.issubdtype(a.dtype, np.floating):
            report.errors.append(f"Depth dtype {a.dtype} is not floating point")
            report.escalate(Status.UNRECOVERABLE)
            return

        if a.shape == (h, w):
            d = a
        elif a.ndim == 1 and a.size == h * w:
            report.errors.append(
                f"Depth is a flat buffer {a.shape}; element count matches ({h},{w})")
            report.actions.append("reshape_depth")
            report.escalate(Status.RECOVERABLE)
            d = a.reshape(h, w)
        else:
            report.errors.append(
                f"Depth shape {a.shape} (size {a.size}) cannot be mapped to ({h},{w}): "
                f"data missing or wrong resolution")
            report.escalate(Status.UNRECOVERABLE)
            return

        finite = np.isfinite(d)
        nan_frac = float(np.mean(np.isnan(d)))
        inf_frac = float(np.mean(np.isinf(d)))
        with np.errstate(invalid="ignore"):
            zero_frac = float(np.mean(d == 0.0))
            oor_frac = float(np.mean(finite & ((d < 0.0) | (d > 1.0))))
        invalid = depth_invalid_mask(d)
        invalid_frac = float(np.mean(invalid))
        report.stats.update(nan_frac=nan_frac, inf_frac=inf_frac,
                            zero_frac=zero_frac, out_of_range_frac=oor_frac,
                            invalid_frac=invalid_frac)

        if invalid_frac > 0:
            report.errors.append(
                f"{invalid_frac*100:.2f}% invalid depth pixels (NaN {nan_frac*100:.2f}%, "
                f"Inf {inf_frac*100:.2f}%, zero {zero_frac*100:.2f}%, "
                f"out-of-range {oor_frac*100:.2f}%)")
            if invalid_frac <= self.max_invalid_fraction:
                report.actions.append("inpaint_depth")
                report.escalate(Status.RECOVERABLE)
            else:
                report.errors.append(
                    f"invalid fraction exceeds the recoverable limit "
                    f"({self.max_invalid_fraction*100:.0f}%): reject instead of inventing data")
                report.escalate(Status.UNRECOVERABLE)

        valid = d[~invalid]
        if valid.size == 0:
            report.errors.append("no valid depth pixels at all")
            report.escalate(Status.UNRECOVERABLE)
            return

        report.stats.update(depth_min=float(valid.min()), depth_max=float(valid.max()),
                            depth_std=float(valid.std()))

        if float(valid.std()) < self.const_std_eps:
            msg = "depth is (almost) constant: broken sensor or renderer?"
            if self.reject_constant_depth:
                report.errors.append(msg)
                report.escalate(Status.UNRECOVERABLE)
            else:
                report.warnings.append(msg)

        if self.roughness_warn is not None:
            rough = depth_roughness(d)
            report.stats["roughness"] = rough
            if np.isfinite(rough) and rough > self.roughness_warn:
                report.warnings.append(
                    f"high-frequency depth noise (roughness {rough:.2e} > "
                    f"{self.roughness_warn:.2e})")
                if self.roughness_reject is not None and rough > self.roughness_reject:
                    report.errors.append(
                        f"noise too high to repair safely (roughness {rough:.2e} > "
                        f"{self.roughness_reject:.2e}): reject")
                    report.escalate(Status.UNRECOVERABLE)
                else:
                    report.actions.append("median_filter")
                    report.escalate(Status.RECOVERABLE)
