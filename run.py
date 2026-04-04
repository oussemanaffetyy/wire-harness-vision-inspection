from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - friendly startup message
    raise SystemExit(
        "Missing Python dependencies. Activate the project virtual environment first with "
        "'source .venv/bin/activate' or run '.venv/bin/python run.py'."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
APP_CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"
ZONES_CONFIG_PATH = PROJECT_ROOT / "config" / "zones.json"
MQTT_CONFIG_PATH = PROJECT_ROOT / "config" / "mqtt.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple launcher for the wire-harness inspection prototype."
    )
    parser.add_argument(
        "preset",
        nargs="?",
        choices=["menu", "offline", "webcam", "esp"],
        default="menu",
        help="Choose a simple execution preset.",
    )
    parser.add_argument(
        "--video",
        help="Optional offline video path override.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Optional webcam index override.",
    )
    parser.add_argument(
        "--stream-url",
        help="Optional ESP/IP stream URL override.",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        help="Optional frame rotation override.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV display window.",
    )
    return parser


def load_app_config() -> dict:
    with APP_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_runtime_args(
    mode: str,
    *,
    video: str | None = None,
    camera: int | None = None,
    stream_url: str | None = None,
    rotation: int | None = None,
    no_display: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        video=video,
        camera=camera,
        stream_url=stream_url,
        rotation=rotation,
        config=str(APP_CONFIG_PATH),
        zones=str(ZONES_CONFIG_PATH),
        mqtt_config=str(MQTT_CONFIG_PATH),
        detector="auto",
        model_path=None,
        no_display=no_display,
    )


def run_inspection(args: argparse.Namespace) -> None:
    from src.app_runner import run_application

    run_application(args)


def run_offline(config: dict, video: str | None, rotation: int | None, no_display: bool) -> None:
    source_cfg = config.get("source", {})
    video_path = video or source_cfg.get("default_offline_video")
    runtime_args = build_runtime_args("offline", video=video_path, rotation=rotation, no_display=no_display)
    run_inspection(runtime_args)


def run_webcam(config: dict, camera: int | None, rotation: int | None, no_display: bool) -> None:
    source_cfg = config.get("source", {})
    camera_index = camera if camera is not None else int(source_cfg.get("default_camera_index", 0))
    runtime_args = build_runtime_args("live", camera=camera_index, rotation=rotation, no_display=no_display)
    run_inspection(runtime_args)


def run_esp_stream(config: dict, stream_url: str | None, rotation: int | None, no_display: bool) -> None:
    source_cfg = config.get("source", {})
    resolved_stream = (stream_url or source_cfg.get("default_stream_url", "")).strip()
    if not resolved_stream:
        raise ValueError(
            "No ESP/IP stream URL configured. Set source.default_stream_url in config/app.yaml or pass --stream-url."
        )
    runtime_args = build_runtime_args("live", stream_url=resolved_stream, rotation=rotation, no_display=no_display)
    run_inspection(runtime_args)


def interactive_menu(config: dict, no_display: bool) -> None:
    source_cfg = config.get("source", {})
    default_video = source_cfg.get("default_offline_video", "data/videos/demo_wire_harness.mp4")
    default_camera = int(source_cfg.get("default_camera_index", 0))
    default_stream = str(source_cfg.get("default_stream_url", "")).strip()

    print()
    print("Prototype d'inspection")
    print("1. Video demo")
    print("2. Webcam PC")
    print("3. Camera ESP / IP stream")
    print("q. Quitter")

    choice = input("Choix: ").strip().lower()
    if choice == "1":
        video_value = input(f"Video [{default_video}]: ").strip() or default_video
        rotation_value = input("Rotation [0/90/180/270, default 0]: ").strip()
        rotation = 0 if not rotation_value else int(rotation_value)
        run_offline(config, video_value, rotation, no_display)
        return

    if choice == "2":
        camera_value = input(f"Index camera [{default_camera}]: ").strip()
        camera_index = default_camera if not camera_value else int(camera_value)
        rotation_value = input("Rotation [0/90/180/270, default 0]: ").strip()
        rotation = 0 if not rotation_value else int(rotation_value)
        run_webcam(config, camera_index, rotation, no_display)
        return

    if choice == "3":
        prompt = "URL stream ESP"
        if default_stream:
            prompt += f" [{default_stream}]"
        prompt += ": "
        stream_value = input(prompt).strip() or default_stream
        if not stream_value:
            raise ValueError("No ESP stream URL provided.")
        rotation_value = input("Rotation [0/90/180/270, default 0]: ").strip()
        rotation = 0 if not rotation_value else int(rotation_value)
        run_esp_stream(config, stream_value, rotation, no_display)
        return

    print("Quit.")


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    config = load_app_config()

    if cli_args.preset == "offline":
        run_offline(config, cli_args.video, cli_args.rotation, cli_args.no_display)
        return

    if cli_args.preset == "webcam":
        run_webcam(config, cli_args.camera, cli_args.rotation, cli_args.no_display)
        return

    if cli_args.preset == "esp":
        run_esp_stream(config, cli_args.stream_url, cli_args.rotation, cli_args.no_display)
        return

    interactive_menu(config, cli_args.no_display)


if __name__ == "__main__":
    main()
