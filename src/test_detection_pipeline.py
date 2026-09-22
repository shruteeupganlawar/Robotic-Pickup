from src.scene_setup import setup_scene
from src.camera import get_camera_image
from src.detect import detect_objects
from src.visualize_detections import draw_detections
import cv2

setup_scene(gui=False)
rgb, depth, _, _, _ = get_camera_image()
detections = detect_objects(rgb)
print(f"Found {len(detections)} objects:", detections)

annotated = draw_detections(rgb, detections)
cv2.imwrite("assets/screenshots/detetction_test.png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print("Saved assets/screenshots/detection_test.png")