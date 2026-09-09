"""Immediate NOK on clip/connector proximity; otherwise OK by default."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.detector.base import Detection, DetectorResult
from .zone_validator import ValidationResult


def box_distance(a: Detection, b: Detection) -> float:
    """Euclidean gap between box edges, zero for overlap or touching edges."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    return math.hypot(max(ax1 - bx2, bx1 - ax2, 0), max(ay1 - by2, by1 - ay2, 0))


def masks_intersect(a: Detection, b: Detection) -> bool | None:
    """None means a mask is unavailable; even a single shared pixel is contact."""
    if a.mask is None or b.mask is None or not a.mask.any() or not b.mask.any():
        return None
    return bool(np.any(a.mask & b.mask))


class ClipAttachmentValidator:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.min_confidence = float(settings.get("min_confidence", 0.35))
        self.connector_near_px = float(settings.get("connector_near_px", 20.0))
        if (not 0 <= self.min_confidence <= 1
                or not math.isfinite(self.connector_near_px) or self.connector_near_px < 0):
            raise ValueError("Invalid clip_attachment thresholds.")

    def validate(self, detector_result: DetectorResult) -> ValidationResult:
        groups: dict[str, list[tuple[int, Detection]]] = {key: [] for key in ("connector", "clip", "cable")}
        shape = None
        for index, detection in enumerate(detector_result.detections):
            if (detection.label not in groups or not math.isfinite(detection.confidence)
                    or detection.confidence < self.min_confidence):
                continue
            x1, y1, x2, y2 = detection.bbox
            if not all(math.isfinite(v) for v in detection.bbox) or x2 <= x1 or y2 <= y1:
                raise ValueError("Detection boxes must have finite coordinates and positive dimensions.")
            if detection.mask is not None:
                if detection.mask.ndim != 2 or detection.mask.dtype != np.bool_:
                    raise ValueError("Clip validation requires full-resolution boolean masks.")
                if shape is not None and shape != detection.mask.shape:
                    raise ValueError("All masks must use the same frame coordinates.")
                shape = detection.mask.shape
            groups[detection.label].append((index, detection))

        missing = [label for label, detections in groups.items() if not detections]
        pairs: list[dict[str, Any]] = []

        def verdict(status: str, detail: str, confidence: float = 0.0) -> ValidationResult:
            return ValidationResult(
                status=status, confidence=round(confidence, 3), missing_classes=missing,
                details=[detail], detector_name=detector_result.detector_name,
                relation={"rule": "clip_attachment", "method": "box_distance_and_mask_intersection",
                          "decision_basis": "connector_contact" if status == "NOK" else "default_ok",
                          "connector_near_px": self.connector_near_px, "pairs": pairs},
            )

        # Absolute priority: no cable evidence can cancel a close connector.
        for clip_index, clip in groups["clip"]:
            for connector_index, connector in groups["connector"]:
                distance = box_distance(clip, connector)
                overlap = masks_intersect(clip, connector)
                pairs.append({"clip_index": clip_index, "target_class": "connector",
                              "target_index": connector_index, "box_gap_px": round(distance, 3),
                              "masks_intersect": overlap})
                if overlap or distance <= self.connector_near_px:
                    return verdict("NOK", f"Clip touche/proche du connecteur : {distance:.1f} px",
                                   min(clip.confidence, connector.confidence))

        # This is the requested default, not visual proof of a correct assembly.
        # Keep confidence at zero rather than invent confidence for missing evidence.
        missing_targets = [label for label in ("clip", "connector") if not groups[label]]
        detail = "OK par defaut : clip eloigne du connecteur"
        if missing_targets:
            detail = "OK par defaut : detection absente : " + ", ".join(missing_targets)
        return verdict("OK", detail)
