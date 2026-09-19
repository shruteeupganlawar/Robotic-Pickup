from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
def detect_objects(rgb_image, conf_threshold=0.15):
    results = model.predict(rgb_image, conf=conf_threshold, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": conf
        })
    return detections

#even if there is only one image, results will come in a list hence it is important to give [0] ie indexing.