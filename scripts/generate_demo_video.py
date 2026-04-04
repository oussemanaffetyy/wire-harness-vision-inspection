from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "videos" / "demo_wire_harness.mp4"


def draw_workstation_frame(frame_index: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (42, 42, 42), dtype=np.uint8)

    cv2.rectangle(frame, (30, 40), (width - 30, height - 40), (70, 70, 70), -1)
    cv2.rectangle(frame, (55, 70), (width - 55, height - 70), (95, 95, 95), -1)

    cv2.putText(
        frame,
        "Synthetic Wire Harness Demo",
        (60, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    zones = [
        ((70, 250), (260, 430), (110, 140, 160)),
        ((1020, 250), (1210, 430), (110, 140, 160)),
        ((560, 120), (740, 300), (110, 140, 160)),
        ((350, 280), (930, 470), (110, 140, 160)),
    ]
    for top_left, bottom_right, color in zones:
        cv2.rectangle(frame, top_left, bottom_right, color, 2)

    offset = int(10 * np.sin(frame_index / 12.0))
    clip_present = not (60 <= frame_index < 90)
    right_connector_misplaced = 120 <= frame_index < 150
    dark_scene = 180 <= frame_index < 205

    cable_points = np.array(
        [
            [170, 340 + offset],
            [380, 350 - offset],
            [610, 315 + offset],
            [840, 350 - offset],
            [1080, 340 + offset],
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [cable_points], False, (0, 215, 255), 18, cv2.LINE_AA)

    left_connector = ((115, 295), (220, 385))
    cv2.rectangle(frame, left_connector[0], left_connector[1], (0, 0, 255), -1)

    if right_connector_misplaced:
        right_connector = ((860, 470), (965, 560))
    else:
        right_connector = ((1060, 295), (1165, 385))
    cv2.rectangle(frame, right_connector[0], right_connector[1], (0, 0, 255), -1)

    if clip_present:
        cv2.circle(frame, (650, 210), 46, (0, 200, 0), -1)

    if dark_scene:
        frame = cv2.convertScaleAbs(frame, alpha=0.58, beta=-20)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)

    status_text = "OK"
    if not clip_present:
        status_text = "NOK - missing clip"
    elif right_connector_misplaced:
        status_text = "NOK - misplaced connector"
    elif dark_scene:
        status_text = "NOK - anomaly"

    cv2.putText(
        frame,
        status_text,
        (60, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frame {frame_index:03d}",
        (width - 220, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def generate_demo_video(output_path: Path, width: int = 1280, height: int = 720, fps: int = 20, total_frames: int = 240) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create demo video at {output_path}")

    try:
        for frame_index in range(total_frames):
            frame = draw_workstation_frame(frame_index, width, height)
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    generate_demo_video(OUTPUT_PATH)
    print(f"Demo video ready: {OUTPUT_PATH}")
