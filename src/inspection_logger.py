from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


class InspectionLogger:
    """Logs inspection results to a single TXT file."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.log_dir = Path("test")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "IACom.txt"

    def log_result(self, status: str, details: str = "") -> None:
        """Log inspection result to the combined file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "OK":
            message = f"[{timestamp}] - Test OK\n"
        elif status == "NOK":
            if details:
                message = f"[{timestamp}] - Test NOK: {details}\n"
            else:
                message = f"[{timestamp}] - Test NOK\n"
        else:
            self.logger.warning("Unknown status: %s", status)
            return

        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(message)
            self.logger.debug("Logged %s to %s", status, self.log_file)
        except Exception as exc:
            self.logger.error("Failed to write to %s: %s", self.log_file, exc)
