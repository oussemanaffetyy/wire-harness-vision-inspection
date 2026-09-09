from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.detector import AnomalyDetector, BaseDetector, MockDetector, YoloDetector
from src.inspection_logger import InspectionLogger
from src.messaging import MqttPublisher
from src.ui_payload import (
    build_event_payload,
    build_metrics_payload,
    build_snapshot_payload,
    build_status_payload,
)
from src.utils import (
    PersonMasker,
    encode_frame_to_base64,
    render_inspection_overlay,
    save_snapshot,
    setup_logger,
)
from src.validation import ClipAttachmentValidator, ZoneValidator
from src.video_source import VideoSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_application(args: Namespace) -> None:
    logger = setup_logger()

    app_config = _load_yaml(_resolve_path(args.config))
    zones_config = _load_json(_resolve_path(args.zones))
    mqtt_config = _load_json(_resolve_path(args.mqtt_config))
    if getattr(args, "no_mqtt", False):
        mqtt_config = {**mqtt_config, "enabled": False}

    zones = zones_config.get("zones", [])
    zones_frame_size = zones_config.get("frame_size", {})
    runtime_cfg = app_config.get("runtime", {})
    source_cfg = app_config.get("source", {})
    detector_cfg = app_config.get("detector", {})
    validation_cfg = app_config.get("validation", {})
    visualization_cfg = app_config.get("visualization", {})
    person_mask_cfg = visualization_cfg.get("person_mask", {})

    mode = args.mode
    show_window = bool(runtime_cfg.get("show_window", True)) and not args.no_display
    max_frames = int(getattr(args, "max_frames", 0))
    if max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    snapshot_dir = _resolve_path(runtime_cfg.get("snapshot_dir", "data/snapshots"))
    snapshot_cooldown_frames = int(runtime_cfg.get("snapshot_cooldown_frames", 20))
    save_nok_snapshots = bool(runtime_cfg.get("save_nok_snapshots", True))
    
    # Keep the Windows inspection journal separate from technical logs in data/logs.
    log_dir = None if sys.platform == "win32" else _resolve_path(runtime_cfg.get("log_dir"))
    video_stream_cooldown = int(runtime_cfg.get("video_stream_cooldown_frames", 5))

    if mode == "offline":
        configured_video = source_cfg.get("default_offline_video")
        requested_videos = getattr(args, "videos", None)
        if requested_videos and args.video:
            raise ValueError("Use --video or --videos, not both.")
        video_paths = [_resolve_path(path) for path in (requested_videos or [args.video or configured_video])]
        frame_rotation = args.rotation if getattr(args, "rotation", None) is not None else int(source_cfg.get("frame_rotation", 0) or 0)
        if any(path is None or not path.is_file() for path in video_paths):
            raise FileNotFoundError(
                "Offline mode requires readable video files. Check --video or source.demo_videos in config/app.yaml."
            )
        source = VideoSource(
            mode="offline",
            video_paths=[str(path) for path in video_paths],
            loop_video=(bool(runtime_cfg.get("loop_video", True)) if getattr(args, "loop", None) is None else args.loop),
            frame_rotation=frame_rotation,
        )
    else:
        stream_url = args.stream_url or source_cfg.get("default_stream_url")
        camera_index = args.camera if args.camera is not None else int(source_cfg.get("default_camera_index", 0))
        frame_rotation = args.rotation if getattr(args, "rotation", None) is not None else int(source_cfg.get("frame_rotation", 0) or 0)
        source = VideoSource(
            mode="live",
            camera_index=camera_index,
            stream_url=stream_url,
            loop_video=False,
            camera_width=source_cfg.get("camera_width"),
            camera_height=source_cfg.get("camera_height"),
            frame_rotation=frame_rotation,
        )

    detector = _create_detector(args, detector_cfg, zones, logger)
    person_masker = _create_person_masker(person_mask_cfg, logger)
    validation_mode = validation_cfg.get("mode", "zones")
    if validation_mode == "clip_attachment":
        model = getattr(detector, "model", None)
        if (getattr(model, "task", None) != "segment"
                or getattr(model, "names", None) != {0: "connector", 1: "clip", 2: "cable"}):
            raise ValueError("clip_attachment requires a segmentation model: connector=0, clip=1, cable=2.")
        if person_masker is not None:
            raise ValueError("Disable person masking for clip_attachment; inference must use the original pixels.")
        validator = ClipAttachmentValidator(validation_cfg.get("clip_attachment", {}))
        zones = []  # The mobile clip is not checked against the old demo table zones.
    elif validation_mode == "zones":
        validator = ZoneValidator(zones=zones, anomaly_threshold=float(validation_cfg.get("anomaly_threshold", 0.55)))
    else:
        raise ValueError(f"Unknown validation mode: {validation_mode}")
    mqtt_publisher = MqttPublisher(mqtt_config, logger)
    inspection_logger = InspectionLogger(log_dir=log_dir, logger=logger)

    logger.info("Starting inspection: mode=%s detector=%s validation=%s display=%s", mode, detector.name, validation_mode, show_window)
    if getattr(detector, "model_path", None):
        logger.info("YOLO model: %s | task=%s | classes=%s", detector.model_path, detector.model.task, detector.model.names)

    recent_times: deque[float] = deque(maxlen=60)
    ok_count = 0
    nok_count = 0
    indeterminate_count = 0
    total_frames = 0
    last_status: str | None = None
    last_snapshot_frame = -snapshot_cooldown_frames
    last_error_label: str | None = None
    last_video_stream_frame = -video_stream_cooldown
    last_source_name: str | None = None

    window_title = "Inspection faisceau - YOLO"
    window_created = False
    try:
        source.open()
        mqtt_publisher.connect()
        while True:
            packet = source.read()
            if packet is None:
                logger.info("End of stream reached.")
                break

            frame = packet.frame
            source_changed = packet.source_name != last_source_name
            if source_changed:
                logger.info("Video source: %s | global frame=%s", packet.source_name, packet.frame_index)
                last_source_name = packet.source_name
            if total_frames and packet.frame_index == 0:
                last_snapshot_frame = -snapshot_cooldown_frames
                last_video_stream_frame = -video_stream_cooldown
            if person_masker is not None:
                frame, _ = person_masker.apply(frame, packet.frame_index)
            active_zones = _scale_zones_to_frame(
                zones,
                source_width=int(zones_frame_size.get("width", frame.shape[1])),
                source_height=int(zones_frame_size.get("height", frame.shape[0])),
                target_width=frame.shape[1],
                target_height=frame.shape[0],
            )
            detector_result = detector.infer(frame)
            validation_result = (validator.validate(detector_result) if isinstance(validator, ClipAttachmentValidator)
                                 else validator.validate(detector_result, zones=active_zones))

            annotated_frame = render_inspection_overlay(
                frame=frame,
                zones=active_zones,
                detections=detector_result.detections,
                result=validation_result,
                mode=mode,
                frame_index=packet.frame_index,
                draw_zones=bool(visualization_cfg.get("draw_zones", True)),
                draw_boxes=bool(visualization_cfg.get("draw_boxes", True)),
                show_labels=bool(visualization_cfg.get("show_labels", True)),
                draw_masks=bool(visualization_cfg.get("draw_masks", True)),
            )

            total_frames += 1
            recent_times.append(time.time())
            fps = _estimate_fps(recent_times)

            if validation_result.status == "OK":
                ok_count += 1
                if not last_error_label:
                    last_error_label = None
            elif validation_result.status == "NOK":
                nok_count += 1
                last_error_label = _derive_error_label(validation_result)
            else:
                indeterminate_count += 1

            snapshot_path: str | None = None
            snapshot_base64: str | None = None
            should_save_snapshot = (
                validation_result.status == "NOK"
                and save_nok_snapshots
                and (packet.frame_index - last_snapshot_frame) >= snapshot_cooldown_frames
            )
            if should_save_snapshot:
                snapshot_path = save_snapshot(annotated_frame, snapshot_dir)
                snapshot_base64 = encode_frame_to_base64(annotated_frame)
                last_snapshot_frame = packet.frame_index
                logger.info("Saved NOK snapshot: %s", snapshot_path)

            status_payload = build_status_payload(
                result=validation_result,
                mode=mode,
                frame_index=packet.frame_index,
                source_name=packet.source_name,
                snapshot_path=snapshot_path,
            )
            metrics_payload = build_metrics_payload(
                mode=mode,
                detector_name=validation_result.detector_name,
                ok_count=ok_count,
                nok_count=nok_count,
                total_frames=total_frames,
                fps=fps,
                last_error_label=last_error_label,
            )
            metrics_payload["indeterminate_count"] = indeterminate_count

            mqtt_publisher.publish("status", status_payload)
            mqtt_publisher.publish("metrics", metrics_payload)
            
            inspection_logger.log_from_validation_result(validation_result)

            should_stream_video = (
                video_stream_cooldown <= 0
                or (packet.frame_index - last_video_stream_frame) >= video_stream_cooldown
            )
            if should_stream_video and mqtt_publisher.enabled:
                frame_base64 = encode_frame_to_base64(annotated_frame)
                if frame_base64:
                    mqtt_publisher.publish("video_stream", {
                        "image_base64": frame_base64,
                        "frame_index": packet.frame_index,
                        "source_name": packet.source_name,
                        "timestamp": status_payload["timestamp"],
                        "status": validation_result.status,
                    })
                    last_video_stream_frame = packet.frame_index
                    logger.debug("Published video frame %s, size: %s bytes", packet.frame_index, len(frame_base64))
                else:
                    logger.warning("Failed to encode frame %s to Base64", packet.frame_index)

            event_type: str | None = None
            if source_changed:
                event_type = "source_change"
            elif validation_result.status != last_status:
                event_type = "status_change"
            elif validation_result.status == "NOK":
                event_type = "nok"

            if event_type:
                event_payload = build_event_payload(
                    result=validation_result,
                    mode=mode,
                    frame_index=packet.frame_index,
                    source_name=packet.source_name,
                    event_type=event_type,
                )
                mqtt_publisher.publish("events", event_payload)
                logger.info(
                    "Inspection event: status=%s frame=%s details=%s",
                    validation_result.status,
                    packet.frame_index,
                    "; ".join(validation_result.details) if validation_result.details else "none",
                )

            if snapshot_path:
                snapshot_payload = build_snapshot_payload(
                    status=validation_result.status,
                    mode=mode,
                    frame_index=packet.frame_index,
                    snapshot_path=snapshot_path,
                    image_base64=snapshot_base64,
                )
                mqtt_publisher.publish("snapshot", snapshot_payload)

            last_status = validation_result.status
            if show_window:
                if not window_created:
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    window_created = True
                    height, width = annotated_frame.shape[:2]
                    scale = min(1.0, 760 / height, 1200 / width)
                    cv2.resizeWindow(window_title, int(width * scale), int(height * scale))
                cv2.imshow(window_title, annotated_frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if max_frames and total_frames >= max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Inspection interrupted by user.")
    finally:
        source.release()
        mqtt_publisher.close()
        if window_created:
            cv2.destroyAllWindows()
        logger.info("Summary: frames=%s OK=%s NOK=%s INDETERMINE=%s", total_frames, ok_count, nok_count, indeterminate_count)
        logger.info("Application shutdown complete.")


def _create_detector(
    args: Namespace,
    detector_cfg: dict[str, Any],
    zones: list[dict[str, Any]],
    logger: Any,
) -> BaseDetector:
    configured_mode = str(detector_cfg.get("mode", "auto")).strip().lower()
    detector_override = args.detector if args.detector != "auto" else configured_mode
    configured_model_path = args.model_path or detector_cfg.get("yolo_model_path")
    resolved_model_path = _resolve_path(configured_model_path) if configured_model_path else None
    anomaly_mode = bool(detector_cfg.get("anomaly_mode", False))

    if detector_override not in {"auto", "mock", "anomaly", "yolo"}:
        logger.warning("Unknown detector mode '%s' in config. Falling back to auto.", detector_override)
        detector_override = "auto"

    if detector_override == "yolo":
        if resolved_model_path is None or not resolved_model_path.exists():
            raise FileNotFoundError("YOLO mode requested, but no valid model path was provided.")
        return YoloDetector(str(resolved_model_path), detector_cfg.get("yolo", {}))

    if detector_override == "anomaly":
        return AnomalyDetector(zones=zones, settings=detector_cfg.get("anomaly", {}))

    if detector_override == "mock":
        return MockDetector()

    if resolved_model_path and resolved_model_path.exists():
        logger.info("Using YOLO detector with model %s", resolved_model_path)
        return YoloDetector(str(resolved_model_path), detector_cfg.get("yolo", {}))

    if anomaly_mode:
        logger.info("Using anomaly detector because anomaly_mode=true and no YOLO model is available.")
        return AnomalyDetector(zones=zones, settings=detector_cfg.get("anomaly", {}))

    logger.info("Using mock detector fallback.")
    return MockDetector()


def _create_person_masker(visualization_cfg: dict[str, Any], logger: Any) -> PersonMasker | None:
    if not bool(visualization_cfg.get("enabled", False)):
        return None

    model_path_value = visualization_cfg.get("model_path", "yolov8n.pt")
    resolved_model_path = _resolve_path(model_path_value)
    if resolved_model_path is None or not resolved_model_path.exists():
        logger.warning("Person masking enabled but model was not found: %s", model_path_value)
        return None

    try:
        logger.info("Using person masking model %s", resolved_model_path)
        return PersonMasker(str(resolved_model_path), visualization_cfg)
    except Exception as exc:
        logger.warning("Person masking disabled: %s", exc)
        return None


def _estimate_fps(recent_times: deque[float]) -> float:
    if len(recent_times) < 2:
        return 0.0
    elapsed = recent_times[-1] - recent_times[0]
    if elapsed <= 0:
        return 0.0
    return (len(recent_times) - 1) / elapsed


def _derive_error_label(result: Any) -> str:
    if getattr(result, "relation", {}).get("rule") == "clip_attachment" and result.status == "NOK":
        return "clip_attached_to_connector"
    if result.anomaly_label:
        return result.anomaly_label
    if result.missing_classes:
        return f"missing:{','.join(result.missing_classes)}"
    if result.misplaced_classes:
        return f"misplaced:{','.join(result.misplaced_classes)}"
    if result.details:
        return result.details[0]
    return "inspection_nok"


def _scale_zones_to_frame(
    zones: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[dict[str, Any]]:
    if source_width <= 0 or source_height <= 0:
        return zones
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    scaled_zones: list[dict[str, Any]] = []
    for zone in zones:
        scaled_zone = dict(zone)
        scaled_zone["polygon"] = [
            [int(round(point[0] * scale_x)), int(round(point[1] * scale_y))]
            for point in zone.get("polygon", [])
        ]
        scaled_zones.append(scaled_zone)
    return scaled_zones


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path
