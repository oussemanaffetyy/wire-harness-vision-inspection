from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class InspectionLogger:
    """Logs inspection results to a single TXT file."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        log_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        if log_dir is not None:
            self.log_dir = Path(log_dir)
        elif sys.platform == "win32":
            self.log_dir = Path("C:/test")
        else:
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

    def log_status(self, status: str, details: str | None = None) -> None:
        """Backward-compatible alias for older callers."""
        self.log_result(status, details or "")

    def log_from_validation_result(self, result: object) -> None:
        """Backward-compatible helper for validation result objects."""
        status = getattr(result, "status", None)
        details = getattr(result, "details", None)
        detail_text = "; ".join(details) if isinstance(details, list) else (details or "")
        if status is not None:
            self.log_result(status, detail_text)
