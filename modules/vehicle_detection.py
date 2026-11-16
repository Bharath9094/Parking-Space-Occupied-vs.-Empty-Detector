from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import MODEL_PATH, VEHICLE_CONFIDENCE

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


@lru_cache(maxsize=1)
def load_model(model_path: str = str(MODEL_PATH)) -> Any:
    path = Path(model_path)
    try:
        from ultralytics import YOLO

        return YOLO(str(path))
    except ImportError as exc:
        raise RuntimeError(
            "The Ultralytics package is missing. Install the project dependencies "
            "with 'pip install -r requirements.txt' and retry."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "The YOLOv8 Nano model could not be loaded or downloaded. "
            "Check your internet connection and install the requirements, then retry."
        ) from exc


def _device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def detect_vehicles(
    image: np.ndarray, confidence: float = VEHICLE_CONFIDENCE
) -> list[dict[str, Any]]:
    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    height, width = image.shape[:2]
    scale = min(1.0, 1280 / max(height, width))
    inference_image = (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else image
    )
    try:
        model = load_model()
        predictions = model.predict(
            source=inference_image,
            conf=confidence,
            classes=list(VEHICLE_CLASSES),
            device=_device(),
            verbose=False,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Vehicle detection failed: {exc}") from exc

    vehicles: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.boxes is None:
            continue
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            if class_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            vehicles.append(
                {
                    "bbox": [
                        max(0, round(x1 / scale)),
                        max(0, round(y1 / scale)),
                        min(width, round(x2 / scale)),
                        min(height, round(y2 / scale)),
                    ],
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": float(box.conf.item()),
                }
            )
    return vehicles
