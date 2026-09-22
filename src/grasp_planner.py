import numpy as np
import pybullet as p

def pixel_to_world(px, py, depth_value, view_matrix, proj_matrix, width, height):
    view_matrix = np.array(view_matrix).reshape(4, 4, order="F")
    proj_matrix = np.array(proj_matrix).reshape(4, 4, order="F")

    x_ndc = (2.0 * px / width) - 1.0
    y_ndc = 1.0 - (2.0 * py / height)
    z_ndc = 2.0 * depth_value - 1.0

    clip_coords = np.array([x_ndc, y_ndc, z_ndc, 1.0])
    inv_proj = np.linalg.inv(proj_matrix)
    inv_view = np.linalg.inv(view_matrix)

    eye_coords = inv_proj @ clip_coords
    eye_coords = eye_coords / eye_coords[3]
    world_coords = inv_view @ eye_coords

    return world_coords[:3]

def get_grasp_point(detection, depth_image, view_matrix, proj_matrix, width, height):
    x1, y1, x2, y2 = detection["bbox"]
    center_px = int((x1 + x2) / 2)
    center_py = int((y1 + y2) / 2)
    center_px = np.clip(center_px, 0, width-1)
    center_py = np.clip(center_py, 0, height-1)

    depth_value = depth_image[center_py, center_px]
    world_point = pixel_to_world(center_px, center_py, depth_value,
                                  view_matrix, proj_matrix, width, height)
    return world_point

def get_grasp_point_from_mask(detection, seg_img, depth_img, view_matrix,
                              proj_matrix, width=640, height=480):
    """Grasp point = centre of the object's TOP surface, from its mask."""
    seg = np.reshape(seg_img, (height, width)).astype(np.int64) & ((1 << 24) - 1)
    ys, xs = np.where(seg == detection["class_id"])
    if len(xs) == 0:
        raise ValueError("No mask pixels for this detection")
    d = depth_img[ys, xs]

    view = np.array(view_matrix).reshape(4, 4, order="F")
    proj = np.array(proj_matrix).reshape(4, 4, order="F")
    inv = np.linalg.inv(proj @ view)

    x_ndc = 2.0 * (xs + 0.5) / width - 1.0
    y_ndc = 1.0 - 2.0 * (ys + 0.5) / height
    z_ndc = 2.0 * d - 1.0
    clip = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)])  # 4 x N
    world = inv @ clip
    world = (world[:3] / world[3]).T                              # N x 3

    top = world[:, 2] > world[:, 2].max() - 0.008   # keep top-face points
    pts = world[top]
    return np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()])