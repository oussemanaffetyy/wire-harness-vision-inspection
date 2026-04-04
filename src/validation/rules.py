from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from src.detector.base import Detection


def point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    polygon_array = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_array, point, False) >= 0


def detections_in_zone(
    detections: Iterable[Detection],
    polygon: list[list[int]],
    label: str | None = None,
    min_confidence: float = 0.0,
) -> list[Detection]:
    zone_detections: list[Detection] = []
    for detection in detections:
        if label and detection.label != label:
            continue
        if detection.confidence < min_confidence:
            continue
        if point_in_polygon(detection.center, polygon):
            zone_detections.append(detection)
    return zone_detections


def detections_by_label(
    detections: Iterable[Detection],
    label: str,
    min_confidence: float = 0.0,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.label == label and detection.confidence >= min_confidence
    ]
