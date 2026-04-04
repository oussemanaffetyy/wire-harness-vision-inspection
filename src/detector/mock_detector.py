from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseDetector, Detection, DetectorResult


@dataclass(slots=True)
class ColorSpec:
    label: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    min_area: int


class MockDetector(BaseDetector):
    name = "mock"

    def __init__(self) -> None:
        self.color_specs = [
            ColorSpec("cable", (15, 80, 80), (40, 255, 255), 1500),
            ColorSpec("clip", (40, 80, 80), (85, 255, 255), 1200),
        ]
        self.red_ranges = [
            ((0, 80, 80), (10, 255, 255)),
            ((170, 80, 80), (180, 255, 255)),
        ]

    def infer(self, frame: np.ndarray) -> DetectorResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for spec in self.color_specs:
            mask = cv2.inRange(hsv, np.array(spec.lower), np.array(spec.upper))
            detections.extend(self._mask_to_detections(mask, spec.label, spec.min_area))

        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.red_ranges:
            red_mask = cv2.bitwise_or(
                red_mask,
                cv2.inRange(hsv, np.array(lower), np.array(upper)),
            )
        detections.extend(self._mask_to_detections(red_mask, "connector", 1800))

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return DetectorResult(
            detections=detections,
            anomaly_score=None,
            anomaly_label=None,
            detector_name=self.name,
            debug={"detections": len(detections)},
        )

    @staticmethod
    def _mask_to_detections(mask: np.ndarray, label: str, min_area: int) -> list[Detection]:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            confidence = max(0.35, min(0.99, 0.45 + (area / 8000.0)))
            detections.append(
                Detection(
                    label=label,
                    confidence=round(confidence, 3),
                    bbox=(x, y, x + w, y + h),
                    center=center,
                    metadata={"area": float(area)},
                )
            )
        return detections
