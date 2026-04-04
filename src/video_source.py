from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    frame_index: int
    source_name: str
    timestamp_ms: float | None = None


class VideoSource:
    def __init__(
        self,
        mode: str,
        video_path: str | None = None,
        camera_index: int = 0,
        stream_url: str | None = None,
        loop_video: bool = True,
        camera_width: int | None = None,
        camera_height: int | None = None,
        frame_rotation: int = 0,
    ) -> None:
        self.mode = mode
        self.video_path = video_path
        self.camera_index = camera_index
        self.stream_url = stream_url
        self.loop_video = loop_video
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame_rotation = frame_rotation
        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0

    def open(self) -> None:
        if self.mode == "offline":
            if not self.video_path:
                raise ValueError("Offline mode requires a video path.")
            self.capture = cv2.VideoCapture(self.video_path)
        else:
            live_source: int | str = self.stream_url if self.stream_url else self.camera_index
            self.capture = cv2.VideoCapture(live_source)
            if not self.stream_url:
                if self.camera_width:
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                if self.camera_height:
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        if not self.capture or not self.capture.isOpened():
            source_name = self._source_name()
            raise RuntimeError(f"Unable to open video source: {source_name}")

    def read(self) -> FramePacket | None:
        if self.capture is None:
            raise RuntimeError("Video source is not open.")

        ok, frame = self.capture.read()
        if not ok:
            if self.mode == "offline" and self.loop_video:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                ok, frame = self.capture.read()
            if not ok:
                return None

        frame = self._apply_rotation(frame)

        timestamp_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC) if self.mode == "offline" else None
        packet = FramePacket(
            frame=frame,
            frame_index=self.frame_index,
            source_name=self._source_name(),
            timestamp_ms=timestamp_ms,
        )
        self.frame_index += 1
        return packet

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _source_name(self) -> str:
        if self.mode == "offline":
            return self.video_path or "offline_video"
        if self.stream_url:
            return self.stream_url
        return f"camera:{self.camera_index}"

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        if self.frame_rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.frame_rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.frame_rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
