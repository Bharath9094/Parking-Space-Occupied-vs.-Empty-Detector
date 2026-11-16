from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "This file is not a valid image or it is corrupted. Please upload a JPG or PNG image."
        ) from exc
    array = np.asarray(image)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return bgr


def image_quality_message(image: np.ndarray) -> str | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(height, width) < 160:
        return "The image is very small; parking-space and vehicle detection may be unreliable."
    if float(np.std(gray)) < 8:
        return "The image has very low contrast; detection may be unreliable."
    return None


def prepare_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def annotate_image(
    image: np.ndarray,
    parking_spaces: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
) -> np.ndarray:
    annotated = image.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 190, 45), 2)
        label = f"{vehicle['class_name']} {vehicle['confidence']:.0%}"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 190, 45),
            2,
            cv2.LINE_AA,
        )
    for space in parking_spaces:
        x1, y1, x2, y2 = space["bbox"]
        is_empty = space["status"] == "EMPTY"
        color = (65, 205, 105) if is_empty else (65, 70, 235)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = space["label"] + (" - EMPTY" if is_empty else " - OCCUPIED")
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(20, y1 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("The analyzed image could not be encoded as PNG.")
    return encoded.tobytes()


def to_pil_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
