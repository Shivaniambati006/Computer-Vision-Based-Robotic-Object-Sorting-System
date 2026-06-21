"""
src/utils/logger.py

Lightweight performance and event logging utility for the sorting
pipeline. Logs both to console and to a rotating log file under the
configured log directory.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "sorting_system",
    log_dir: str = "logs/",
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure and return a logger writing to console + rotating file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        # Avoid duplicate handlers if setup_logger is called multiple times.
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "sorting_system.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class PerformanceLogger:
    """
    Periodically logs throughput and detection performance metrics
    at a configurable interval, without blocking the main pipeline loop.
    """

    def __init__(self, logger: logging.Logger, interval_sec: float = 5.0):
        self.logger = logger
        self.interval_sec = interval_sec
        self._last_log_time = time.time()

    def maybe_log(self, sorted_count: int, rate_per_min: float, fps: float):
        now = time.time()
        if now - self._last_log_time >= self.interval_sec:
            self.logger.info(
                f"Performance — Sorted: {sorted_count} | "
                f"Rate: {rate_per_min:.2f}/min | FPS: {fps:.1f}"
            )
            self._last_log_time = now
