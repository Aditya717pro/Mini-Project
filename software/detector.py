from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Find best.pt automatically from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best.pt"


# Load trained YOLO model
model = YOLO(str(MODEL_PATH))


def process_frame(frame, target_object, confidence_threshold=0.5):
    """
    Process one camera frame.

    Input:
        frame          -> OpenCV/Numpy image
        target_object  -> Pen / Spectacles / Keys

    Output:
        Dictionary containing detection and position information.
    """

    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    image_height, image_width = result.orig_shape

    # Divide image into three horizontal regions
    left_boundary = image_width / 3
    right_boundary = 2 * image_width / 3

    best_detection = None

    # Check all detected objects
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Only consider requested target
        if class_name != target_object:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detection = {
            "object": class_name,
            "confidence": confidence,
            "bbox": (
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ),
            "center": (
                round(center_x),
                round(center_y)
            )
        }

        # Keep the highest-confidence target
        if (
            best_detection is None
            or confidence > best_detection["confidence"]
        ):
            best_detection = detection

    # Target not found
    if best_detection is None:

        return {
            "status": "TARGET_NOT_FOUND",
            "target": target_object,
            "position": None,
            "command": "SEARCH"
        }, result

    center_x = best_detection["center"][0]

    # Determine horizontal position
    if center_x < left_boundary:
        position = "LEFT"
        command = "TURN_LEFT"

    elif center_x > right_boundary:
        position = "RIGHT"
        command = "TURN_RIGHT"

    else:
        position = "CENTER"
        command = "FORWARD"

    best_detection["status"] = "TARGET_FOUND"
    best_detection["target"] = target_object
    best_detection["position"] = position
    best_detection["command"] = command

    return best_detection, result


def load_image(image_path):
    """Load an image using OpenCV."""

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    return frame


def display_result(result):
    """Display YOLO annotated result without cv2.imshow()."""

    from PIL import Image

    annotated = result.plot()

    # Convert OpenCV BGR image to RGB
    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    # Display using the default Windows image viewer
    Image.fromarray(annotated_rgb).show()