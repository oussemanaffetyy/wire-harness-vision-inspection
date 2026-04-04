from __future__ import annotations

from typing import Any

from src.validation.zone_validator import ValidationResult

from src.utils.timestamps import utc_now_iso


def build_status_payload(
    result: ValidationResult,
    mode: str,
    frame_index: int,
    source_name: str,
    snapshot_path: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "status": result.status,
        "mode": mode,
        "detector": result.detector_name,
        "confidence": result.confidence,
        "frame_index": frame_index,
        "source_name": source_name,
        "missing_classes": result.missing_classes,
        "misplaced_classes": result.misplaced_classes,
        "anomaly_score": result.anomaly_score,
        "anomaly_label": result.anomaly_label,
        "matched_zone_ids": result.matched_zone_ids,
        "failed_zone_ids": result.failed_zone_ids,
        "details": result.details,
        "snapshot_path": snapshot_path,
    }


def build_metrics_payload(
    mode: str,
    detector_name: str,
    ok_count: int,
    nok_count: int,
    total_frames: int,
    fps: float,
    last_error_label: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "mode": mode,
        "detector": detector_name,
        "ok_count": ok_count,
        "nok_count": nok_count,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "last_error_label": last_error_label,
    }


def build_event_payload(
    result: ValidationResult,
    mode: str,
    frame_index: int,
    source_name: str,
    event_type: str,
) -> dict[str, Any]:
    message_parts = result.details or [f"status={result.status}"]
    return {
        "timestamp": utc_now_iso(),
        "event_type": event_type,
        "status": result.status,
        "mode": mode,
        "detector": result.detector_name,
        "frame_index": frame_index,
        "source_name": source_name,
        "message": " | ".join(message_parts),
        "missing_classes": result.missing_classes,
        "misplaced_classes": result.misplaced_classes,
        "anomaly_score": result.anomaly_score,
        "anomaly_label": result.anomaly_label,
    }


def build_snapshot_payload(
    status: str,
    mode: str,
    frame_index: int,
    snapshot_path: str,
    image_base64: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "status": status,
        "mode": mode,
        "frame_index": frame_index,
        "snapshot_path": snapshot_path,
        "image_base64": image_base64,
    }
