from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectorResult:
    detections: list[Detection]
    anomaly_score: float | None = None
    anomaly_label: str | None = None
    detector_name: str = "unknown"
    debug: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def infer(self, frame: np.ndarray) -> DetectorResult:
        raise NotImplementedError
