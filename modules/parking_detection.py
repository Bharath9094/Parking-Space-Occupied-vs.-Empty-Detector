from typing import Any

import cv2
import numpy as np


def _intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def detect_parking_regions(image: np.ndarray) -> list[dict[str, Any]]:
    """Return conservative rectangular parking-space proposals for user review."""
    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(height, width) < 160 or float(np.std(gray)) < 8:
        return []

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = height * width
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.003 * image_area <= area <= 0.12 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        if box_width < 24 or box_height < 24:
            continue
        aspect_ratio = box_width / box_height
        if not 0.35 <= aspect_ratio <= 2.85:
            continue
        rectangularity = area / max(1, box_width * box_height)
        if rectangularity < 0.48:
            continue
        normalized_area = area / image_area
        area_score = max(0.0, 1.0 - abs(normalized_area - 0.025) / 0.10)
        shape_score = min(1.0, rectangularity)
        confidence = min(0.88, 0.38 + 0.30 * shape_score + 0.20 * area_score)
        if confidence < 0.68:
            continue
        candidates.append(
            {
                "bbox": (x, y, x + box_width, y + box_height),
                "confidence": confidence,
                "source": "automatic proposal",
            }
        )

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        box = candidate["bbox"]
        if any(
            _intersection_over_union(box, previous["bbox"]) > 0.72
            for previous in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 80:
            break
    return selected


def regions_from_normalized(
    normalized_regions: list[dict[str, float]], width: int, height: int
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for item in normalized_regions:
        x1 = round(float(item["x1"]) * width)
        y1 = round(float(item["y1"]) * height)
        x2 = round(float(item["x2"]) * width)
        y2 = round(float(item["y2"]) * height)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 > x1 and y2 > y1:
            regions.append({"bbox": (x1, y1, x2, y2), "confidence": None})
    return regions


def normalize_regions(
    regions: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, float]]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    normalized = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        normalized.append(
            {
                "x1": max(0.0, min(1.0, x1 / width)),
                "y1": max(0.0, min(1.0, y1 / height)),
                "x2": max(0.0, min(1.0, x2 / width)),
                "y2": max(0.0, min(1.0, y2 / height)),
            }
        )
    return normalized


def classify_regions(
    regions: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    classified = []
    for index, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region["bbox"]
        slot_area = max(1, (x2 - x1) * (y2 - y1))
        matches = []
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            intersection = max(0, min(x2, vx2) - max(x1, vx1)) * max(
                0, min(y2, vy2) - max(y1, vy1)
            )
            overlap = intersection / slot_area
            if overlap >= overlap_threshold:
                matches.append(vehicle)
        classified.append(
            {
                **region,
                "label": f"Space {index}",
                "status": "OCCUPIED" if matches else "EMPTY",
                "vehicle_confidence": (
                    max(item["confidence"] for item in matches) if matches else None
                ),
                "vehicle_class": (
                    matches[0]["class_name"] if matches else None
                ),
            }
        )
    return classified
