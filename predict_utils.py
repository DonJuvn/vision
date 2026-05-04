"""
predict_utils.py — Cross-platform prediction utilities for PPE detection
Provides reusable functions for image prediction with proper error handling
and headless operation (no plt.show() dependency).
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO


# ── Configuration ────────────────────────────────────────────────────────
# Platform-specific GPU handling: only disable on Windows if needed
if sys.platform == "win32":
    # Windows may have GPU issues; set environment variable before importing
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CONF_THRESH = 0.35

# BGR color tuples for OpenCV (both cases supported for flexibility)
CLASS_COLORS = {
    "Head":   (60, 60, 255),      # Red (BGR) — no helmet
    "Helmet": (60, 200, 60),      # Green — has helmet
    "Person": (255, 180, 60),     # Blue-ish — person
    "head":   (60, 60, 255),
    "helmet": (60, 200, 60),
    "person": (255, 180, 60),
}

# Default color for unknown classes
DEFAULT_COLOR = (200, 200, 0)


# ── Validation Functions ─────────────────────────────────────────────────

def validate_image_path(img_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file exists and is readable.

    Args:
        img_path: Path to the image file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(img_path)

    # Check if path exists
    if not path.exists():
        return False, f"File does not exist: {img_path}"

    # Check if it's a file (not a directory)
    if not path.is_file():
        return False, f"Path is not a file: {img_path}"

    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid image format: {path.suffix}. Supported: {valid_extensions}"

    # Try to read the file to verify it's a valid image
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False, f"Cannot read image file (may be corrupted): {img_path}"
    except Exception as e:
        return False, f"Error reading image: {str(e)}"

    return True, None


def validate_model_path(model_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a model file exists and has a valid extension.

    Args:
        model_path: Path to the model file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(model_path)

    if not path.exists():
        return False, f"Model file does not exist: {model_path}"

    if not path.is_file():
        return False, f"Model path is not a file: {model_path}"

    valid_extensions = {'.pt', '.onnx', '.engine'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid model format: {path.suffix}. Supported: {valid_extensions}"

    return True, None


# ── Model Loading ─────────────────────────────────────────────────────────

def load_model(model_path: str = "ppe_best.pt") -> Optional[YOLO]:
    """
    Load a YOLO model with proper error handling.

    Args:
        model_path: Path to the model file

    Returns:
        YOLO model instance or None if loading failed
    """
    # Validate model path
    is_valid, error_msg = validate_model_path(model_path)
    if not is_valid:
        print(f"[ERROR] {error_msg}")
        return None

    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        print(f"Classes: {list(model.names.values())}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return None


# ── Detection Drawing ─────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                   model_names: Dict[int, str]) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image using OpenCV.

    Args:
        image: Input image (numpy array, BGR format)
        detections: List of detection dictionaries with keys:
                   - bbox: [x1, y1, x2, y2]
                   - class: class name string
                   - confidence: float value
        model_names: Mapping of class IDs to class names

    Returns:
        Image with drawn detections (BGR format)
    """
    img_copy = image.copy()

    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        class_name = det.get('class', 'Unknown')
        confidence = det.get('confidence', 0.0)

        # Get color for this class
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_copy, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

        # Draw label text
        cv2.putText(img_copy, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_copy


def draw_status_panel(image: np.ndarray, counts: Dict[str, int],
                     violation: bool) -> np.ndarray:
    """
    Draw a status panel on the image showing detection counts and violation status.

    Args:
        image: Input image (numpy array, BGR format)
        counts: Dictionary mapping class names to counts
        violation: Boolean indicating if there's a violation (no helmet)

    Returns:
        Image with status panel (BGR format)
    """
    img_copy = image.copy()
    H, W = img_copy.shape[:2]

    # Prepare status message
    no_helmet = counts.get("Head", 0) + counts.get("head", 0)
    if violation:
        msg = f"VIOLATION: {no_helmet} without helmet!"
        msg_color = (0, 0, 220)  # Red (BGR)
    else:
        msg = "ALL WEARING HELMETS"
        msg_color = (0, 180, 0)  # Green (BGR)

    # Draw semi-transparent overlay
    overlay = img_copy.copy()
    panel_height = 70
    cv2.rectangle(overlay, (0, 0), (W, panel_height), (30, 30, 30), -1)
    img_copy = cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0)

    # Draw status message
    cv2.putText(img_copy, msg, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)

    # Draw counts (handle both cases)
    helmet_n = counts.get('Helmet', 0) + counts.get('helmet', 0)
    head_n = counts.get('Head', 0) + counts.get('head', 0)
    person_n = counts.get('Person', 0) + counts.get('person', 0)
    stats = (f"Helmet: {helmet_n}  |  "
             f"Head (no helmet): {head_n}  |  "
             f"Person: {person_n}")
    cv2.putText(img_copy, stats, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return img_copy


# ── Main Prediction Function ──────────────────────────────────────────────

def predict_image(img_path: str, model: YOLO,
                 output_path: Optional[str] = None,
                 draw_boxes: bool = True) -> Dict[str, Any]:
    """
    Run PPE detection on an image and return structured results.

    Args:
        img_path: Path to the input image
        model: Loaded YOLO model
        output_path: Optional path to save annotated image
        draw_boxes: Whether to draw bounding boxes on saved image

    Returns:
        Dictionary with keys:
        - success: bool - whether prediction succeeded
        - error: str - error message if success=False
        - detections: list of dict with keys:
            - class: str (class name)
            - confidence: float
            - bbox: [x1, y1, x2, y2]
        - counts: dict mapping class names to counts
        - violation: bool - True if someone without helmet detected
        - output_path: str - path to saved image (if output_path provided)
    """
    result = {
        'success': False,
        'error': None,
        'detections': [],
        'counts': {},
        'violation': False,
        'output_path': None
    }

    # Validate image path
    is_valid, error_msg = validate_image_path(img_path)
    if not is_valid:
        result['error'] = error_msg
        return result

    # Read image
    try:
        img = cv2.imread(img_path)
        if img is None:
            result['error'] = f"Failed to read image: {img_path}"
            return result
    except Exception as e:
        result['error'] = f"Error reading image: {str(e)}"
        return result

    # Run prediction
    try:
        results = model.predict(img_path, conf=CONF_THRESH, verbose=False)
    except Exception as e:
        result['error'] = f"Prediction failed: {str(e)}"
        return result

    # Process results
    detections = []
    counts = {}
    class_names = model.names

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf_val = float(box.conf[0])

            detections.append({
                'class': cls_name,
                'confidence': conf_val,
                'bbox': [x1, y1, x2, y2]
            })

            counts[cls_name] = counts.get(cls_name, 0) + 1

    # Determine violation (someone without helmet - handle both cases)
    violation = counts.get("Head", 0) > 0 or counts.get("head", 0) > 0

    # Save annotated image if requested
    if output_path and draw_boxes:
        try:
            # Draw detections and status
            annotated = draw_detections(img, detections, class_names)
            annotated = draw_status_panel(annotated, counts, violation)

            # Create output directory if needed
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            cv2.imwrite(str(out_path), annotated)
            result['output_path'] = str(out_path)
        except Exception as e:
            result['error'] = f"Failed to save image: {str(e)}"
            # Still return results even if saving failed

    result['success'] = True
    result['detections'] = detections
    result['counts'] = counts
    result['violation'] = violation

    return result


# ── Convenience Functions ─────────────────────────────────────────────────

def get_violation_status(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Get a human-readable violation status from prediction result.

    Args:
        result: Prediction result dictionary from predict_image()

    Returns:
        Tuple of (is_violation, status_message)
    """
    if not result.get('success'):
        return False, "Prediction failed"

    counts = result.get('counts', {})
    no_helmet = counts.get("Head", 0) + counts.get("head", 0)

    if no_helmet > 0:
        return True, f"VIOLATION: {no_helmet} person(s) without helmet"
    return False, "All persons wearing helmets"
