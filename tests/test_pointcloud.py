import numpy as np
import pytest

from safe_rgbd import (DepthConverter, camera_to_world, depth_to_camera_points,
                       intrinsics_from_proj, pixels_to_world)
from src.grasp_planner import pixel_to_world       # the demo's original projection code


def test_intrinsics_from_60deg_fov(proj_matrix, hw):
    fx, fy, cx, cy = intrinsics_from_proj(proj_matrix, hw[1], hw[0])
    assert fy == pytest.approx(240 / np.tan(np.radians(30)), rel=1e-4)
    assert fx == pytest.approx(fy, rel=1e-4)            # square pixels
    assert (cx, cy) == (320.0, 240.0)


def test_centre_pixel_on_optical_axis(proj_matrix, hw):
    intr = intrinsics_from_proj(proj_matrix, hw[1], hw[0])
    z = np.full(hw, 1.5)
    pts = depth_to_camera_points(z, *intr)
    assert pts.shape == (*hw, 3)
    assert np.allclose(pts[239, 319, :2], 0.0, atol=0.01)   # ~ centre of image
    assert np.all(pts[..., 2] == 1.5)


def test_matches_demo_pixel_to_world(proj_matrix, view_matrix, hw):
    conv = DepthConverter.from_proj_matrix(proj_matrix)
    intr = intrinsics_from_proj(proj_matrix, hw[1], hw[0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = int(rng.integers(0, hw[1]))
        v = int(rng.integers(0, hw[0]))
        d = float(rng.uniform(0.9, 0.99))
        old = pixel_to_world(u + 0.5, v + 0.5, d, view_matrix, proj_matrix, hw[1], hw[0])
        z = conv.buffer_to_metric(np.array([d]))
        new = pixels_to_world([u], [v], z, view_matrix, *intr)[0]
        assert np.allclose(old, new, atol=1e-3)


def test_camera_to_world_shape(view_matrix):
    pts = np.zeros((4, 5, 3))
    assert camera_to_world(pts, view_matrix).shape == (4, 5, 3)
