from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.detector.base import Detection
from src.validation.zone_validator import ValidationResult


DETECTION_COLORS = {
    "cable": (70, 170, 60),
    "clip": (220, 120, 40),
    "connector": (60, 60, 220),
}


def render_inspection_overlay(
    frame: np.ndarray,
    zones: list[dict[str, Any]],
    detections: list[Detection],
    result: ValidationResult,
    mode: str,
    frame_index: int,
    draw_zones: bool = True,
    draw_boxes: bool = True,
    show_labels: bool = True,
    draw_masks: bool = True,
) -> np.ndarray:
    annotated = frame.copy()
    if draw_masks:
        _draw_masks(annotated, detections)
    if draw_zones:
        _draw_zones(annotated, zones, result.failed_zone_ids)
    _draw_detections(annotated, detections, draw_boxes, show_labels)

    # Put the verdict outside the camera image so no clip or mask is hidden.
    annotated = cv2.copyMakeBorder(annotated, 64, 48, 0, 0, cv2.BORDER_CONSTANT, value=(28, 32, 36))
    color = {"OK": (40, 170, 65), "NOK": (45, 50, 220)}.get(result.status, (135, 135, 135))
    cv2.rectangle(annotated, (0, 0), (5, 63), color, -1)
    qualifier = "STATUT ESTIME" if result.relation else "STATUT"
    _text(annotated, f"{qualifier} : {result.status}", (14, 27), 0.65, color, 2)
    _text(annotated, "Connecteur rouge | Clip bleu | Cable vert", (14, 50), 0.40, (215, 215, 215))
    note = result.details[0] if result.details else "Inspection en cours"
    _text(annotated, note, (12, annotated.shape[0] - 28), 0.43, (240, 240, 240))
    _text(annotated, "Q ou Echap : quitter", (12, annotated.shape[0] - 9), 0.36, (175, 175, 175))
    return annotated


def _text(frame: np.ndarray, label: str, origin: tuple[int, int], scale: float,
          color: tuple[int, int, int], thickness: int = 1) -> None:
    width = frame.shape[1] - origin[0] - 10
    text = label
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > width:
        text = text[:-1]
    if text != label and len(text) > 3:
        text = text[:-3] + "..."
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_masks(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        mask = detection.mask
        if mask is None:
            continue
        if mask.shape != frame.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("Display masks must align with the original frame.")
        color = DETECTION_COLORS.get(detection.label, (180, 180, 180))
        frame[mask] = (frame[mask] * 0.65 + np.asarray(color) * 0.35).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 1)


def _draw_zones(frame: np.ndarray, zones: list[dict[str, Any]], failed_zone_ids: list[str]) -> None:
    for zone in zones:
        points = np.array(zone["polygon"], dtype=np.int32)
        color = (40, 40, 220) if zone.get("id") in failed_zone_ids else (255, 180, 0)
        cv2.polylines(frame, [points], True, color, 2)
        x, y, _, _ = cv2.boundingRect(points)
        _text(frame, f"{zone.get('id', 'zone')} | {zone.get('required_class', '-')}",
              (x, max(20, y - 10)), 0.5, color)


def _draw_detections(frame: np.ndarray, detections: list[Detection],
                     draw_boxes: bool, show_labels: bool) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = DETECTION_COLORS.get(detection.label, (180, 180, 180))
        if draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        if show_labels:
            _text(frame, f"{detection.label} {detection.confidence:.2f}",
                  (max(0, x1), max(14, y1 - 5)), 0.42, color)
