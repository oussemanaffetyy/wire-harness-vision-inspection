from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import BaseDetector, Detection, DetectorResult


class AnomalyDetector(BaseDetector):
    name = "anomaly"

    def __init__(self, zones: list[dict[str, Any]], settings: dict[str, Any]) -> None:
        self.zones = zones
        self.threshold = float(settings.get("threshold", 0.55))
        self.blur_threshold = float(settings.get("blur_threshold", 60.0))
        self.darkness_threshold = float(settings.get("darkness_threshold", 55.0))
        self.min_foreground_ratio = float(settings.get("min_foreground_ratio", 0.01))

    def infer(self, frame: np.ndarray) -> DetectorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())

        detections: list[Detection] = []
        zone_penalties: list[float] = []
        zone_occupancy: dict[str, float] = {}

        for zone in self.zones:
            occupancy = self._compute_zone_occupancy(gray, zone["polygon"])
            zone_occupancy[zone["id"]] = occupancy

            if occupancy >= self.min_foreground_ratio:
                x, y, x2, y2 = self._bounding_box(zone["polygon"])
                confidence = max(0.2, min(0.99, occupancy / max(self.min_foreground_ratio, 1e-6)))
                detections.append(
                    Detection(
                        label=zone.get("required_class", "zone_object"),
                        confidence=round(min(confidence, 0.99), 3),
                        bbox=(x, y, x2, y2),
                        center=((x + x2) // 2, (y + y2) // 2),
                        metadata={"source": "zone_occupancy", "zone_id": zone["id"]},
                    )
                )
                zone_penalties.append(0.0)
            else:
                deficit = 1.0 - (occupancy / max(self.min_foreground_ratio, 1e-6))
                zone_penalties.append(max(0.0, min(1.0, deficit)))

        blur_penalty = 0.0 if blur_value >= self.blur_threshold else 1.0 - (blur_value / max(self.blur_threshold, 1e-6))
        darkness_penalty = 0.0 if brightness >= self.darkness_threshold else 1.0 - (brightness / max(self.darkness_threshold, 1e-6))
        anomaly_score = float(np.mean(zone_penalties + [blur_penalty, darkness_penalty]))

        anomaly_label = None
        if anomaly_score >= self.threshold:
            if blur_penalty > 0.35:
                anomaly_label = "blurred_scene"
            elif darkness_penalty > 0.35:
                anomaly_label = "dark_scene"
            else:
                anomaly_label = "low_zone_occupancy"

        return DetectorResult(
            detections=detections,
            anomaly_score=round(anomaly_score, 3),
            anomaly_label=anomaly_label,
            detector_name=self.name,
            debug={
                "blur_value": round(float(blur_value), 3),
                "brightness": round(brightness, 3),
                "zone_occupancy": {key: round(value, 4) for key, value in zone_occupancy.items()},
            },
        )

    def _compute_zone_occupancy(self, gray_frame: np.ndarray, polygon: list[list[int]]) -> float:
        mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
        masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        edges = cv2.Canny(masked, 50, 150)

        active_pixels = int(np.count_nonzero(edges))
        total_pixels = int(np.count_nonzero(mask))
        if total_pixels == 0:
            return 0.0
        return active_pixels / total_pixels

    @staticmethod
    def _bounding_box(polygon: list[list[int]]) -> tuple[int, int, int, int]:
        points = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        return x, y, x + w, y + h
