from __future__ import annotations

from datetime import datetime
from pathlib import Path


class InspectionResultLogger:
    def __init__(self, log_dir: str | Path = "test", file_name: str = "IACom.txt") -> None:
        self.log_path = Path(log_dir) / file_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_result(self, status: str, details: str = "") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if status == "OK":
            line = f"[{timestamp}] Test OK\n"
        elif status == "NOK":
            suffix = f": {details}" if details else ""
            line = f"[{timestamp}] Test NOK{suffix}\n"
        else:
            return

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
