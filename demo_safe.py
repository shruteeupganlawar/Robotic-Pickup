"""Perception-to-grasp demo with the safe RGB-D layer in front of the planner.

Examples (run from the project root):
    python demo_safe.py                                       # clean frame
    python demo_safe.py --corrupt nan --severity 0.10         # repaired, then picked
    python demo_safe.py --corrupt zero_region --severity 0.10 --on-target   # rejected: no motion
    python demo_safe.py --corrupt truncate_depth --severity 0.10            # rejected: no motion
    python demo_safe.py --gui                                 # watch it in the PyBullet window
"""

import argparse

import numpy as np
import pybullet as p

from safe_rgbd import SafePerception, localize_object
from safe_rgbd.corruption import CORRUPTION_NAMES, apply_corruption
from src.camera import get_camera_image
from src.detect_seg import detect_objects_by_segmentation
from src.execute_grasp import execute_pick
from src.scene_setup import setup_scene

H, W = 480, 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--corrupt", default="clean", choices=["clean"] + CORRUPTION_NAMES)
    ap.add_argument("--severity", type=float, default=0.10)
    ap.add_argument("--on-target", action="store_true",
                    help="centre a zero_region dropout on the cube")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scene = setup_scene(gui=args.gui)
    for _ in range(240):                       # let the objects settle
        p.stepSimulation()

    print("Step 1: capture")
    rgb, depth, view, proj, seg = get_camera_image()      # rgb is (H, W, 3): alpha stripped
    cube_id = scene["objects"]["cube"]
    cube_mask = (seg.astype(np.int64) & 0xFFFFFF) == cube_id

    print(f"Step 2: corrupt ({args.corrupt}, severity {args.severity})")
    centre = None
    if args.on_target:
        ys, xs = np.nonzero(cube_mask)
        centre = (xs.mean(), ys.mean())
    rng = np.random.default_rng(args.seed)
    rgb_c, depth_c = apply_corruption(args.corrupt, rgb, depth, args.severity, rng, centre)

    print("Step 3: safe perception")
    sp = SafePerception(H, W, proj, rgb_channels=3)
    frame = sp.process(rgb_c, depth_c)
    print(frame.report.summary())
    if not frame.accepted:
        print(f"\nFRAME REJECTED -> the arm will NOT move.\nReason: {frame.reason}")
        return
    print(f"Actions taken: {frame.actions or 'none'}   ({frame.latency_ms:.1f} ms)")

    print("Step 4: detect + localize")
    dets = detect_objects_by_segmentation(
        seg, list(scene["objects"].values()), list(scene["objects"].keys()))
    print("  detections:", [d["class_name"] for d in dets])
    loc = localize_object(cube_mask, frame.depth_metric, frame.confidence, view,
                          sp.intrinsics, min_confidence=args.min_confidence)
    if not loc.ok:
        print(f"\nTARGET REJECTED -> the arm will NOT move.\nReason: {loc.reason}")
        return
    truth = np.array(p.getBasePositionAndOrientation(cube_id)[0])
    print(f"  grasp point {np.round(loc.point, 3)}  (confidence {loc.confidence:.2f}); "
          f"true cube xy error {np.linalg.norm(loc.point[:2] - truth[:2]) * 100:.1f} cm")

    print("Step 5: pick")
    execute_pick(scene["robot"], loc.point)
    z = p.getBasePositionAndOrientation(cube_id)[0][2]
    print(f"\nCube height after pick: {z:.3f} m -> {'SUCCESS' if z > 0.15 else 'FAILED'}")


if __name__ == "__main__":
    main()
