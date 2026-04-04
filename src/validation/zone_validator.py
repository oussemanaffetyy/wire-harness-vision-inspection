from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.detector.base import DetectorResult

from .rules import detections_by_label, detections_in_zone


@dataclass(slots=True)
class ValidationResult:
    status: str
    confidence: float
    matched_zone_ids: list[str] = field(default_factory=list)
    failed_zone_ids: list[str] = field(default_factory=list)
    missing_classes: list[str] = field(default_factory=list)
    misplaced_classes: list[str] = field(default_factory=list)
    anomaly_score: float | None = None
    anomaly_label: str | None = None
    details: list[str] = field(default_factory=list)
    detector_name: str = "unknown"


class ZoneValidator:
    def __init__(self, zones: list[dict[str, Any]], anomaly_threshold: float = 0.55) -> None:
        self.zones = zones
        self.anomaly_threshold = anomaly_threshold

    def validate(
        self,
        detector_result: DetectorResult,
        zones: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        validation_zones = zones if zones is not None else self.zones
        matched_zone_ids: list[str] = []
        failed_zone_ids: list[str] = []
        missing_classes: list[str] = []
        misplaced_classes: list[str] = []
        details: list[str] = []
        confidence_values: list[float] = []

        for zone in validation_zones:
            zone_id = zone.get("id", "zone")
            required_class = zone.get("required_class")
            min_confidence = float(zone.get("min_confidence", 0.0))
            rule_type = zone.get("rule_type", "required_in_zone")
            polygon = zone["polygon"]

            matches = detections_in_zone(
                detector_result.detections,
                polygon=polygon,
                label=required_class,
                min_confidence=min_confidence,
            )

            if rule_type == "required_in_zone":
                if matches:
                    best_match = max(matches, key=lambda item: item.confidence)
                    matched_zone_ids.append(zone_id)
                    confidence_values.append(best_match.confidence)
                else:
                    failed_zone_ids.append(zone_id)
                    if required_class:
                        missing_classes.append(required_class)
                        if detections_by_label(detector_result.detections, required_class, min_confidence):
                            misplaced_classes.append(required_class)
                    details.append(f"{zone_id}: missing {required_class}")

            elif rule_type == "forbidden_in_zone":
                if matches:
                    failed_zone_ids.append(zone_id)
                    if required_class:
                        misplaced_classes.append(required_class)
                    details.append(f"{zone_id}: forbidden {required_class} present")
                else:
                    matched_zone_ids.append(zone_id)

            else:
                if matches:
                    best_match = max(matches, key=lambda item: item.confidence)
                    matched_zone_ids.append(zone_id)
                    confidence_values.append(best_match.confidence)

        anomaly_score = detector_result.anomaly_score
        anomaly_label = detector_result.anomaly_label
        if anomaly_score is not None and anomaly_score >= self.anomaly_threshold:
            details.append(f"anomaly score {anomaly_score:.3f}")
            if anomaly_label:
                details.append(f"anomaly label {anomaly_label}")

        status = "NOK" if failed_zone_ids or (anomaly_score is not None and anomaly_score >= self.anomaly_threshold) else "OK"

        if confidence_values:
            confidence = sum(confidence_values) / len(confidence_values)
        elif anomaly_score is not None:
            confidence = max(0.0, 1.0 - anomaly_score)
        else:
            confidence = 0.0

        return ValidationResult(
            status=status,
            confidence=round(confidence, 3),
            matched_zone_ids=self._unique(matched_zone_ids),
            failed_zone_ids=self._unique(failed_zone_ids),
            missing_classes=self._unique(missing_classes),
            misplaced_classes=self._unique(misplaced_classes),
            anomaly_score=anomaly_score,
            anomaly_label=anomaly_label,
            details=self._unique(details),
            detector_name=detector_result.detector_name,
        )

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                unique_values.append(value)
        return unique_values
