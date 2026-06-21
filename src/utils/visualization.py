"""
src/utils/visualization.py

Debug visualization helpers: draws bounding boxes, centroids, color
labels, and an FPS counter onto frames for pipeline verification.
"""

import time
from typing import List

import cv2

from src.vision.detection import DetectedObject


COLOR_DRAW_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}
DEFAULT_DRAW_COLOR = (255, 255, 255)


class FPSCounter:
    """Simple rolling FPS counter for overlay display."""

    def __init__(self, smoothing: float = 0.9):
        self._last_time = time.time()
        self._fps = 0.0
        self.smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = (
                self.smoothing * self._fps + (1 - self.smoothing) * instant_fps
                if self._fps > 0
                else instant_fps
            )
        return self._fps


def draw_detections(frame, detections: List[DetectedObject]):
    """Overlay bounding boxes, centroids, and color labels on a frame."""
    for obj in detections:
        x, y, w, h = obj.bounding_box
        cx, cy = obj.centroid
        draw_color = COLOR_DRAW_MAP.get(obj.color, DEFAULT_DRAW_COLOR)

        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        cv2.circle(frame, (cx, cy), 5, draw_color, -1)
        cv2.putText(
            frame,
            f"{obj.color} ({cx},{cy})",
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_color,
            2,
        )
    return frame


def draw_fps(frame, fps: float):
    """Overlay current FPS value on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return frame


def draw_sorting_rate(frame, rate_per_min: float, sorted_count: int):
    """Overlay current sorting throughput stats on the frame."""
    cv2.putText(
        frame,
        f"Sorted: {sorted_count}  |  Rate: {rate_per_min:.1f}/min",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    return frame
