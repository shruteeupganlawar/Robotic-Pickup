import numpy as np

SEVERITIES = [0.01, 0.05, 0.10, 0.25, 0.50]

def inject_nan(depth, fraction, rng):
    out = np.array(depth, dtype=np.float32, copy=True)
    n = max(int(out.size * fraction), 1)
    idx = rng.choice(out.size, n, replace=False)
    out.flat[idx] = np.nan
    return out

def inject_inf(depth, fraction, rng):
    out = np.array(depth, dtype=np.float32, copy=True)
    n = max(int(out.size * fraction), 1)
    idx = rng.choice(out.size, n, replace=False)
    out.flat[idx] = np.inf
    return out

def zero_region(depth, area_fraction, rng, center=None):
    out = np.array(depth, dtype=np.float32, copy=True)
    h, w = out.shape
    scale = np.sqrt(area_fraction)
    rh, rw = max(int(h * scale), 1), max(int(w * scale), 1)
    if center is None:
        cu = int(rng.integers(rw // 2, max(w - rw // 2, rw // 2 + 1)))
        cv = int(rng.integers(rh // 2, max(h - rh // 2, rh // 2 + 1)))
    else:
        cu, cv = int(center[0]), int(center[1])
    u1, u2 = max(cu - rw // 2, 0), min(cu + rw // 2, w)
    v1, v2 = max(cv - rh // 2, 0), min(cv + rh // 2, h)
    out[v1:v2, u1:u2] = 0.0
    return out

def constant_depth(depth, value=0.5):
    return np.full(np.shape(depth), value, dtype=np.float32)

def out_of_range(depth, fraction, rng):
    out = np.array(depth, dtype=np.float32, copy=True)
    n = max(int(out.size * fraction), 1)
    idx = rng.choice(out.size, n, replace=False)
    out.flat[idx] = rng.choice([-0.3, 1.5], size=n).astype(np.float32)
    return out

def gaussian_noise(depth, severity, rng):
    out = np.array(depth, dtype=np.float32, copy=True)
    out += rng.normal(0.0, severity * 0.01, size=out.shape).astype(np.float32)
    return np.clip(out, 1e-6, 1.0)

def depth_bias(depth, severity):
    out = np.array(depth, dtype=np.float32, copy=True) - np.float32(severity * 0.01)
    return np.clip(out, 1e-6, 1.0)

def flatten(data):
    return np.asarray(data).flatten()


def truncate(data, fraction):
    flat = np.asarray(data).flatten()
    n = max(int(flat.size * fraction), 1)
    return flat[:-n].copy()

CORRUPTION_NAMES = [
    "nan", "inf", "zero_region", "out_of_range", "noise", "depth_bias", "constant_depth",
    "flatten_rgb", "flatten_depth", "truncate_rgb", "truncate_depth", "mixed",
]


def apply_corruption(name, rgb, depth, severity, rng, center=None):
    if name == "clean":
        return rgb, depth
    if name == "nan":
        return rgb, inject_nan(depth, severity, rng)
    if name == "inf":
        return rgb, inject_inf(depth, severity, rng)
    if name == "zero_region":
        return rgb, zero_region(depth, severity, rng, center)
    if name == "out_of_range":
        return rgb, out_of_range(depth, severity, rng)
    if name == "noise":
        return rgb, gaussian_noise(depth, severity, rng)
    if name == "depth_bias":
        return rgb, depth_bias(depth, severity)
    if name == "constant_depth":
        return rgb, constant_depth(depth)
    if name == "flatten_rgb":
        return flatten(rgb), depth
    if name == "flatten_depth":
        return rgb, flatten(depth)
    if name == "truncate_rgb":
        return truncate(rgb, severity), depth
    if name == "truncate_depth":
        return rgb, truncate(depth, severity)
    if name == "mixed":
        d = inject_nan(depth, severity / 2, rng)
        d = zero_region(d, severity / 2, rng, center)
        d = gaussian_noise(d, severity, rng)
        return rgb, d
    raise ValueError(f"Unknown corruption: {name}")
