"""Pinhole geometry: metric depth -> 3D points.

Camera-frame convention used here: x right, y DOWN, z FORWARD (standard pinhole).
PyBullet's view matrix uses the OpenGL convention (x right, y up, camera looks along -z),
so camera_to_world() converts between them.
"""

import numpy as np


def intrinsics_from_proj(proj_matrix, width, height):
    """(fx, fy, cx, cy) in pixels from a PyBullet column-major projection matrix."""
    m = np.asarray(proj_matrix, dtype=np.float64).ravel()
    fx = m[0] * width / 2.0
    fy = m[5] * height / 2.0
    return fx, fy, width / 2.0, height / 2.0


def depth_to_camera_points(depth_metric, fx, fy, cx, cy):
    """(H, W) metric depth -> (H, W, 3) points in the pinhole camera frame."""
    z = np.asarray(depth_metric, dtype=np.float64)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)   # pixel centres
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def camera_to_world(points, view_matrix):
    """Pinhole-camera points (..., 3) -> world points (..., 3)."""
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    gl = np.stack([flat[:, 0], -flat[:, 1], -flat[:, 2], np.ones(len(flat))], axis=1)
    inv_view = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F"))
    world = gl @ inv_view.T
    return world[:, :3].reshape(pts.shape)


def pixels_to_world(us, vs, z, view_matrix, fx, fy, cx, cy):
    """Selected pixels (column us, row vs) with metric depth z -> (N, 3) world points."""
    us = np.asarray(us, dtype=np.float64) + 0.5
    vs = np.asarray(vs, dtype=np.float64) + 0.5
    z = np.asarray(z, dtype=np.float64)
    cam = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)
    return camera_to_world(cam, view_matrix)
