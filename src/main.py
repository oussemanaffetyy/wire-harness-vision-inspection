from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wire harness / cable assembly inspection prototype."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        required=True,
        help="Use an offline video file or a live webcam.",
    )
    parser.add_argument(
        "--video",
        help="Path to an offline video file. Required for offline mode unless a default is configured.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index for live mode. Defaults to the configured camera index.",
    )
    parser.add_argument(
        "--stream-url",
        help="Optional network stream URL for live mode, for example an ESP32-CAM MJPEG stream.",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        help="Optional frame rotation applied before inference.",
    )
    parser.add_argument(
        "--config",
        default="config/app.yaml",
        help="Path to the main YAML configuration file.",
    )
    parser.add_argument(
        "--zones",
        default="config/zones.json",
        help="Path to the zone definition JSON file.",
    )
    parser.add_argument(
        "--mqtt-config",
        default="config/mqtt.json",
        help="Path to the MQTT JSON configuration file.",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "mock", "anomaly", "yolo"],
        default="auto",
        help="Detector selection override.",
    )
    parser.add_argument(
        "--model-path",
        help="Optional YOLO model path. If the file exists, YOLO mode is used automatically.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV display window.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.app_runner import run_application

    run_application(args)


if __name__ == "__main__":
    main()
