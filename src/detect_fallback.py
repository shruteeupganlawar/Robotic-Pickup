import cv2
import numpy as np

def detect_objects_by_color(rgb_image):
    """Simple color-thresholding detector as a guaranteed fallback
    for synthetic simulation ojects with distinct colors."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    detections = []

    lower = np.array([0, 70, 50])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({
            "bbox": (x, y, x+w, y+h),
            "class_id": 0,
            "class_name": "object",
            "confidence": 1.0
        })
    return detections