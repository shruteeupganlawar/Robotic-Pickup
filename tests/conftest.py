import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H, W = 480, 640
NEAR, FAR = 0.1, 3.0

@pytest.fixture
def hw():
    return H, W

@pytest.fixture
def proj_matrix():
    return p.computeProjectionMatrixFOV(60, W / H, NEAR, FAR)

@pytest.fixture
def view_matrix():
    return p.computeViewMatrixFromYawPitchRoll([0.5, 0, 0.1], 1.0, 50, -35, 0, 2)

@pytest.fixture
def clean_rgb():
    return np.random.default_rng(0).integers(0, 256, (H, W, 4), dtype=np.uint8)

@pytest.fixture
def clean_depth():
    d = np.linspace(0.90, 0.99, W, dtype=np.float32)[None, :]
    d= d+ np.linspace(0.0, 0.005, H, dtype=np.float32)[:, None]
    d[200:300, 250:350] = 0.93
    return d.astype(np.float32)