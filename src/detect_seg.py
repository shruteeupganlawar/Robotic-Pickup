# src/detect_seg.py
import numpy as np
import pybullet as p

def detect_objects_by_segmentation(seg_img, object_ids, names, width=640, height=480):
    seg = np.reshape(seg_img, (height, width)).astype(np.int64)
    seg = seg & ((1 << 24) - 1)   # strip link-index bits, keep body id
    detections = []
    for name, body_id in zip(names, object_ids):
        ys, xs = np.where(seg == body_id)
        if len(xs) < 20:
            continue
        detections.append({
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "class_id": body_id,
            "class_name": name,
            "confidence": 1.0,
        })
    return detections