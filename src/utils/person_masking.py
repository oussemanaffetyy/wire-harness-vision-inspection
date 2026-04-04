from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detector.base import Detection

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional until YOLO deps are installed
    YOLO = None


class PersonMasker:
    def __init__(self, model_path: str, settings: dict[str, Any]) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required for person masking.")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Person masking model not found: {model_file}")

        self.model = YOLO(str(model_file))
        self.confidence_threshold = float(settings.get("confidence_threshold", 0.35))
        self.image_size = int(settings.get("image_size", 640))
        self.device = settings.get("device", "cpu")
        self.every_n_frames = max(1, int(settings.get("every_n_frames", 3)))
        self.overlay_alpha = float(settings.get("overlay_alpha", 0.55))
        self.overlay_color = tuple(int(v) for v in settings.get("overlay_color_bgr", [70, 190, 90]))
        self.blur_kernel = max(1, int(settings.get("blur_kernel", 31)))
        self.min_area_ratio = float(settings.get("min_area_ratio", 0.01))
        self._cached_detections: list[Detection] = []
        self._last_refresh_frame = -1

    def apply(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, list[Detection]]:
        if self._should_refresh(frame_index):
            self._cached_detections = self._detect_people(frame)
            self._last_refresh_frame = frame_index

        masked = frame.copy()
        if not self._cached_detections:
            return masked, []

        overlay = masked.copy()
        for detection in self._cached_detections:
            x1, y1, x2, y2 = detection.bbox
            x1 = max(0, min(masked.shape[1] - 1, x1))
            y1 = max(0, min(masked.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(masked.shape[1], x2))
            y2 = max(y1 + 1, min(masked.shape[0], y2))

            roi = masked[y1:y2, x1:x2]
            if roi.size:
                kernel = self._valid_kernel(min(roi.shape[0], roi.shape[1]))
                if kernel > 1:
                    masked[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.overlay_color, -1)

        cv2.addWeighted(overlay, self.overlay_alpha, masked, 1.0 - self.overlay_alpha, 0, masked)
        return masked, list(self._cached_detections)

    def _detect_people(self, frame: np.ndarray) -> list[Detection]:
        predictions = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        prediction = predictions[0]
        names = prediction.names if hasattr(prediction, "names") else {}

        detections: list[Detection] = []
        for box in prediction.boxes:
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            if label != "person":
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            detections.append(
                Detection(
                    label="person",
                    confidence=round(confidence, 3),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    metadata={"source": "person_masker"},
                )
            )

        frame_area = float(frame.shape[0] * frame.shape[1])
        min_area = frame_area * self.min_area_ratio
        filtered = [
            detection
            for detection in detections
            if (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]) >= min_area
        ]
        filtered.sort(key=lambda detection: detection.confidence, reverse=True)

        deduped: list[Detection] = []
        for detection in filtered:
            if any(self._bbox_iou(detection.bbox, kept.bbox) > 0.45 for kept in deduped):
                continue
            deduped.append(detection)
        return deduped

    def _should_refresh(self, frame_index: int) -> bool:
        return self._last_refresh_frame < 0 or (frame_index - self._last_refresh_frame) >= self.every_n_frames

    def _valid_kernel(self, min_side: int) -> int:
        kernel = min(self.blur_kernel, max(1, min_side - 1))
        if kernel % 2 == 0:
            kernel -= 1
        return max(1, kernel)

    @staticmethod
    def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union
