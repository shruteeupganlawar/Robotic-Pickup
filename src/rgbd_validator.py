import numpy as np

class FrameValidationError(Exception):
    pass

def validate_frame(rgb, depth):
    """Basic checks before trusting a frame for downstream planning. """
    if rgb is None or depth is None:
        raise FrameValidationError("Frame is None")

    if np.isnan(depth).any() or np.isinf(depth).any():
        raise FrameValidationError("Depth contains NaN/Inf Values")

    if depth.shape[0]==0 or depth.shape[1]==0:
        raise FrameValidationError("Depth has zero size dimension")

    valid_ratio = np.sum((depth>0) & (depth < 1)) / depth.size
    if valid_ratio < 0.1:
        raise FrameValidationError(f"Only {valid_ratio*100:.1f}% of depth pixels are valid")
    return True

def inject_corruption(depth, mode="nan"):
    corrupted = depth.copy()
    if mode == "nan":
        corrupted[100:150, 100:150] = np.nan
    elif mode == "truncate":
        corrupted = corrupted[:corrupted.shape[0]//2]
    elif mode == "zero":
        corrupted[:] = 0
    return corrupted