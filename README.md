# AI-Powered Assistive Object Retrieval Robot

An intelligent mobile robot that detects, locates and retrieves commonly used objects using computer vision and robotic control.

## Machine Learning Module

The ML module uses YOLOv8n for object detection.

### Detected Objects

| Class ID | Object |
|---|---|
| 0 | Pen |
| 1 | Spectacles |
| 2 | Keys |

### Dataset

A balanced dataset of 1500 images was used:

- Pen: 500 images
- Spectacles: 500 images
- Keys: 500 images

Dataset split:

- Training: 1200 images
- Validation: 150 images
- Testing: 150 images

### Training

- Model: YOLOv8n
- Epochs: 50
- Image Size: 640 × 640
- Training Platform: Google Colab
- GPU: NVIDIA Tesla T4

### Test Results

- Precision: 80.2%
- Recall: 79.9%
- mAP@50: 82.5%
- mAP@50-95: 51.1%

### Files

- `Model_Training.ipynb` — Google Colab training and testing workflow
- `best.pt` — trained YOLOv8n model
- `data.yaml` — dataset configuration

## Next Development Stage

The next stage is to integrate the trained object detection model with the ESP32-CAM and robot controller.

Expected communication flow:

ESP32-CAM → Image Capture → YOLOv8 → Object Detection → Target Position → ESP32 → Motor Control

The ESP32-CAM integration and robot movement/navigation modules will be developed next.
