from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.detector.base import Detection
from src.validation.zone_validator import ValidationResult


DETECTION_COLORS = {
    "cable": (0, 215, 255),
    "clip": (0, 200, 0),
    "connector": (0, 0, 255),
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
) -> np.ndarray:
    annotated = frame.copy()

    if draw_zones:
        _draw_zones(annotated, zones, result.failed_zone_ids)
    if draw_boxes:
        _draw_detections(annotated, detections)

    _draw_status_banner(annotated, result, mode, frame_index)
    if show_labels:
        _draw_footer(annotated, result)
    return annotated


def _draw_zones(frame: np.ndarray, zones: list[dict[str, Any]], failed_zone_ids: list[str]) -> None:
    for zone in zones:
        points = np.array(zone["polygon"], dtype=np.int32)
        color = (40, 40, 220) if zone.get("id") in failed_zone_ids else (255, 180, 0)
        cv2.polylines(frame, [points], True, color, 2)

        x, y, w, h = cv2.boundingRect(points)
        label = f"{zone.get('id', 'zone')} | {zone.get('required_class', '-')}"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = DETECTION_COLORS.get(detection.label, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_status_banner(frame: np.ndarray, result: ValidationResult, mode: str, frame_index: int) -> None:
    status_color = (30, 180, 30) if result.status == "OK" else (0, 0, 220)
    banner_right = min(frame.shape[1] - 20, 640)
    small_frame = frame.shape[1] < 700
    line1_scale = 0.72 if small_frame else 1.0
    line2_scale = 0.48 if small_frame else 0.62
    line1_y = 56 if small_frame else 58
    line2_y = 84 if small_frame else 92
    banner_bottom = 96 if small_frame else 110
    cv2.rectangle(frame, (20, 20), (banner_right, banner_bottom), status_color, -1)
    cv2.putText(
        frame,
        f"STATUS: {result.status}",
        (35, line1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        line1_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    detail_text = f"mode={mode} detector={result.detector_name} frame={frame_index} conf={result.confidence:.2f}"
    if small_frame and len(detail_text) > 40:
        detail_text = f"{detail_text[:37]}..."
    cv2.putText(
        frame,
        detail_text,
        (35, line2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        line2_scale,
        (255, 255, 255),
        1 if small_frame else 2,
        cv2.LINE_AA,
    )


def _draw_footer(frame: np.ndarray, result: ValidationResult) -> None:
    detail_text = " | ".join(result.details[:3]) if result.details else "inspection running"
    max_chars = 55 if frame.shape[1] < 700 else 110
    if len(detail_text) > max_chars:
        detail_text = f"{detail_text[:max_chars - 3]}..."

    footer_y = frame.shape[0] - 24
    cv2.rectangle(frame, (20, footer_y - 26), (frame.shape[1] - 20, footer_y + 10), (20, 20, 20), -1)
    cv2.putText(
        frame,
        detail_text,
        (30, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
