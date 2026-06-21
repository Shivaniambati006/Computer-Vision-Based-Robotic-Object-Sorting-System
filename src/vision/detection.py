"""
src/vision/detection.py

Contour detection and geometric object localization stage.

Given a cleaned binary mask for a color class, this module extracts
valid object contours, filters them by minimum area, and computes
bounding boxes and centroids for downstream coordinate mapping.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class DetectedObject:
    """Represents a single localized object detected in a frame."""

    color: str
    centroid: tuple          # (cx, cy) in pixel coordinates
    bounding_box: tuple      # (x, y, w, h)
    area: float
    contour: np.ndarray


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Extract external contours from a binary mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def compute_centroid(contour: np.ndarray) -> tuple:
    """
    Compute the centroid (cx, cy) of a contour using image moments.

    Falls back to the bounding box center if the contour area is
    degenerate (zero moment), which can occur with very thin/noisy
    contours that survive area filtering.
    """
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w // 2, y + h // 2)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy)


def localize_objects(
    mask: np.ndarray, color: str, min_contour_area: float
) -> List[DetectedObject]:
    """
    Full localization routine for a single color mask:
        find contours -> filter by area -> compute bounding box + centroid.
    """
    detected = []
    for contour in find_contours(mask):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        bbox = cv2.boundingRect(contour)  # (x, y, w, h)
        centroid = compute_centroid(contour)

        detected.append(
            DetectedObject(
                color=color,
                centroid=centroid,
                bounding_box=bbox,
                area=area,
                contour=contour,
            )
        )
    return detected


def localize_all(
    masks: dict, min_contour_area: float
) -> List[DetectedObject]:
    """
    Run localization across all color masks produced by the
    preprocessing stage. Returns a flat list of DetectedObject
    instances across all color classes, ready for sorting logic.
    """
    all_objects: List[DetectedObject] = []
    for color, mask in masks.items():
        all_objects.extend(localize_objects(mask, color, min_contour_area))
    return all_objects
