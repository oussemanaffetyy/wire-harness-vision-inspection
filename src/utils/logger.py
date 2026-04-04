from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(log_dir: str | Path = "data/logs", file_name: str = "inspection.log") -> logging.Logger:
    logger = logging.getLogger("wire_harness_inspection")
    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        Path(log_dir) / file_name,
        maxBytes=1_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
