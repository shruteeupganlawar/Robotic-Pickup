# src/main_demo.py
import time
import cv2
import pybullet as p

from src.scene_setup import setup_scene
from src.camera import get_camera_image
from src.detect_seg import detect_objects_by_segmentation
from src.visualize_detections import draw_detections
from src.rgbd_validator import validate_frame, FrameValidationError
from src.grasp_planner import get_grasp_point_from_mask
from src.execute_grasp import execute_pick


def run_demo(record=False):
    scene = setup_scene(gui=True)
    robot_id = scene["robot"]

    log_id = None
    if record:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                     "assets/videos/demo.mp4")

    for _ in range(240):
        p.stepSimulation()

    print("Step 1: Capturing camera frame...")
    rgb, depth, view_matrix, proj_matrix, seg = get_camera_image()

    print("Step 2: Validating frame...")
    try:
        validate_frame(rgb, depth)
        print("  Frame validated successfully.")
    except FrameValidationError as e:
        print(f"  Frame REJECTED: {e}. Aborting pick.")
        _stop_recording(log_id)
        return

    print("Step 3: Running object detection (segmentation mask)...")
    object_names = list(scene["objects"].keys())
    object_ids = list(scene["objects"].values())
    detections = detect_objects_by_segmentation(seg, object_ids, object_names)
    print(f"  Found {len(detections)} object(s).")
    for d in detections:
        print(f"    - {d['class_name']}: bbox={d['bbox']}")
    if not detections:
        print("  No objects detected. Aborting.")
        _stop_recording(log_id)
        return

    annotated = draw_detections(rgb, detections)
    cv2.imwrite("assets/screenshots/demo_detection.png",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print("Step 4: Computing grasp point for first detection...")
    target_det = detections[0]
    grasp_point = get_grasp_point_from_mask(target_det, seg, depth, view_matrix,
                                  proj_matrix, 640, 480)
    print(f"  Target: {target_det['class_name']}")
    print(f"  Grasp point: {grasp_point}")

    print("Step 5: Executing pick sequence...")
    execute_pick(robot_id, grasp_point)

    print("\nDemo complete. Letting simulation settle for viewing...")
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1 / 240)

    cube_pos, _ = p.getBasePositionAndOrientation(scene["objects"]["cube"])
    print("Cube final position:", cube_pos)

    _stop_recording(log_id)


def _stop_recording(log_id):
    if log_id is not None:
        p.stopStateLogging(log_id)
        print("Video saved to assets/videos/demo.mp4")


if __name__ == "__main__":
    run_demo()