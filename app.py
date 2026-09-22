import gradio as gr
import numpy as np
from PIL import Image
from src.detect import detect_objects
from src.visualize_detections import draw_detections
import cv2

def process_image(uploaded_image):
    rgb = np.array(uploaded_image.convert("RGB"))
    detections = detect_objects(rgb)
    annotated = draw_detections(rgb, detections)

    summary = f"Found {len(detections)} object(s):\n"
    for det in detections:
        summary += f"  - {det['class_name']} (confidence: {det['confidence']:.2f})\n"

    return Image.fromarray(annotated), summary

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(label="Detections"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Perception-to-Grasp Demo: Object Detection Stage",
    description=(
        "Upload any image to see object detection results. "
        "This is the perception front-end of a full RGB-D validation "
        "→ detection → grasp-planning pipeline built for robotic manipulation."
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
