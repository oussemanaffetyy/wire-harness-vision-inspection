from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class InspectionLogger:
    """Logs inspection results to separate OK and NOK text files."""

    def __init__(self, log_dir: str | Path | None = None, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

        if log_dir is None:
            is_windows = sys.platform == "win32"
            if is_windows:
                self.log_dir = Path("C:/Test/TA_com_Txt")
            else:
                self.log_dir = Path.home() / "Test" / "TA_com_Txt"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ok_file = self.log_dir / "Test_OK.txt"
        self.nok_file = self.log_dir / "Test_NOK.txt"

        self.logger.info("Inspection logger initialized at %s", self.log_dir)

    def log_status(self, status: str, details: str | None = None) -> None:
        """Log inspection result to appropriate file."""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if status == "OK":
            message = f"[{timestamp}] - OK\n"
            target_file = self.ok_file
        elif status == "NOK":
            if details:
                message = f"[{timestamp}] - NOK: {details}\n"
            else:
                message = f"[{timestamp}] - NOK\n"
            target_file = self.nok_file
        else:
            self.logger.warning("Unknown status: %s", status)
            return

        try:
            with target_file.open("a", encoding="utf-8") as f:
                f.write(message)
            self.logger.debug("Logged %s to %s", status, target_file)
        except Exception as exc:
            self.logger.error("Failed to write to %s: %s", target_file, exc)

    def log_from_validation_result(self, result: Any) -> None:
        """Log inspection result from validation result object."""
        status = result.status

        if status == "NOK":
            error_label = None
            if result.anomaly_label:
                error_label = result.anomaly_label
            elif result.missing_classes:
                error_label = f"missing:{','.join(result.missing_classes)}"
            elif result.misplaced_classes:
                error_label = f"misplaced:{','.join(result.misplaced_classes)}"
            elif result.details:
                error_label = result.details[0]

            self.log_status("NOK", error_label)
        else:
            self.log_status("OK")

    def get_log_dir(self) -> Path:
        """Return the log directory path."""
        return self.log_dir
