"""Object localization from a mask + metric depth + confidence."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pointcloud import pixels_to_world


@dataclass
class Localization:
    ok: bool
    point: Optional[np.ndarray] = None   # world-frame top-surface centre
    confidence: float = 0.0
    n_pixels: int = 0
    reason: str = ""


def localize_object(mask, depth_metric, confidence, view_matrix, intrinsics,
                    min_confidence=0.5, top_band=0.008):
    """World-frame grasp target = median of the object's top-surface points.

    Refuses (ok=False) when the depth under the mask is not trustworthy enough.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return Localization(False, reason="empty object mask")

    z = depth_metric[ys, xs]
    c = confidence[ys, xs]
    finite = np.isfinite(z)
    if not finite.any():
        return Localization(False, n_pixels=int(xs.size), reason="no finite depth under mask")

    mean_conf = float(c[finite].mean())
    if mean_conf < min_confidence:
        return Localization(False, confidence=mean_conf, n_pixels=int(xs.size),
                            reason=f"depth confidence {mean_conf:.2f} < {min_confidence:.2f}")

    pts = pixels_to_world(xs[finite], ys[finite], z[finite], view_matrix, *intrinsics)
    top = pts[:, 2] > pts[:, 2].max() - top_band
    point = np.median(pts[top], axis=0)
    return Localization(True, point, mean_conf, int(xs.size))
