from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseDetector, Detection, DetectorResult

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dependency may be absent until installed
    YOLO = None


class YoloDetector(BaseDetector):
    name = "yolo"

    def __init__(self, model_path: str, settings: dict[str, Any]) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install requirements.txt first.")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_file}")

        self.model = YOLO(str(model_file))
        self.confidence_threshold = float(settings.get("confidence_threshold", 0.35))
        self.image_size = int(settings.get("image_size", 640))
        self.device = settings.get("device", "cpu")

    def infer(self, frame: np.ndarray) -> DetectorResult:
        predictions = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        prediction = predictions[0]

        detections: list[Detection] = []
        names = prediction.names if hasattr(prediction, "names") else {}

        for box in prediction.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            detections.append(
                Detection(
                    label=label,
                    confidence=round(confidence, 3),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    metadata={"class_id": cls_id},
                )
            )

        return DetectorResult(
            detections=detections,
            anomaly_score=None,
            anomaly_label=None,
            detector_name=self.name,
            debug={"detections": len(detections)},
        )
