from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .timestamps import timestamp_for_filename


def save_snapshot(frame: np.ndarray, output_dir: str | Path, prefix: str = "nok") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{prefix}_{timestamp_for_filename()}.jpg"
    if not cv2.imwrite(str(file_path), frame):
        raise RuntimeError(f"Unable to save snapshot: {file_path}")
    return str(file_path)


def encode_frame_to_base64(frame: np.ndarray, quality: int = 80) -> str | None:
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")
