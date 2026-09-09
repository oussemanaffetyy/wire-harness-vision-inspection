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
        video_paths: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.video_path = video_path
        if video_path and video_paths:
            raise ValueError("Use video_path or video_paths, not both.")
        self.video_paths = list(video_paths or ([video_path] if video_path else []))
        self.video_number = 0
        self.source_frame_index = 0
        self.camera_index = camera_index
        self.stream_url = stream_url
        self.loop_video = loop_video
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame_rotation = frame_rotation
        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0

    def open(self) -> None:
        self.release()
        self.frame_index = 0
        self.video_number = 0
        self.source_frame_index = 0
        if self.mode == "offline":
            if not self.video_paths:
                raise ValueError("Offline mode requires a video path.")
            self.capture = cv2.VideoCapture(self.video_paths[0])
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
            self.release()
            raise RuntimeError(f"Unable to open video source: {source_name}")

    def read(self) -> FramePacket | None:
        if self.capture is None:
            raise RuntimeError("Video source is not open.")

        while True:
            ok, frame = self.capture.read()
            if ok:
                break
            if self.mode != "offline":
                return None
            if self.source_frame_index == 0:
                raise RuntimeError(f"Video contains no readable frames: {self._source_name()}")
            next_video = self.video_number + 1
            if next_video == len(self.video_paths):
                if not self.loop_video:
                    return None
                next_video = 0
            # Switch capture, not application: model, MQTT and counters stay alive.
            self.release()
            self.video_number = next_video
            self.source_frame_index = 0
            self.capture = cv2.VideoCapture(self.video_paths[self.video_number])
            if not self.capture.isOpened():
                source_name = self._source_name()
                self.release()
                raise RuntimeError(f"Unable to open video source: {source_name}")

        frame = self._apply_rotation(frame)

        timestamp_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC) if self.mode == "offline" else None
        packet = FramePacket(
            frame=frame,
            frame_index=self.frame_index,
            source_name=self._source_name(),
            timestamp_ms=timestamp_ms,
        )
        self.frame_index += 1
        self.source_frame_index += 1
        return packet

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _source_name(self) -> str:
        if self.mode == "offline":
            return self.video_paths[self.video_number] if self.video_paths else "offline_video"
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
