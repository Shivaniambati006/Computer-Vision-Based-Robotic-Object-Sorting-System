"""
src/vision/preprocessing.py

Frame preprocessing stage of the vision pipeline:
    BGR -> HSV conversion -> color masking/segmentation -> morphological
    noise removal.

This module is intentionally stateless per-frame; all tunable parameters
are loaded from config/color_thresholds.yaml and config/system_config.yaml.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Convert a BGR frame (OpenCV's default capture format) to HSV.

    HSV is preferred over BGR/RGB for color segmentation because it
    decouples chromaticity (Hue) from brightness (Value), making
    thresholding far more robust to lighting variation, shadows, and
    reflections common in industrial environments.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def build_color_mask(hsv_frame: np.ndarray, color_def: Dict) -> np.ndarray:
    """
    Build a binary mask isolating pixels within a given color class's
    HSV range(s). Supports colors requiring two ranges (e.g., red, which
    wraps around the hue circle at 0/179).
    """
    lower1 = np.array(color_def["lower1"], dtype=np.uint8)
    upper1 = np.array(color_def["upper1"], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower1, upper1)

    if "lower2" in color_def and "upper2" in color_def:
        lower2 = np.array(color_def["lower2"], dtype=np.uint8)
        upper2 = np.array(color_def["upper2"], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    return mask


def clean_mask(mask: np.ndarray, morph_cfg: Dict) -> np.ndarray:
    """
    Apply morphological operations to remove noise from a binary mask.

    Erosion strips away small isolated noise blobs; dilation restores
    and slightly expands the surviving object regions, closing small
    gaps left by segmentation artifacts.
    """
    erosion_kernel = np.ones(tuple(morph_cfg["erosion_kernel"]), np.uint8)
    dilation_kernel = np.ones(tuple(morph_cfg["dilation_kernel"]), np.uint8)

    cleaned = cv2.erode(
        mask, erosion_kernel, iterations=morph_cfg["erosion_iterations"]
    )
    cleaned = cv2.dilate(
        cleaned, dilation_kernel, iterations=morph_cfg["dilation_iterations"]
    )
    return cleaned


def segment_all_colors(
    frame: np.ndarray, color_classes: Dict, morph_cfg: Dict
) -> Dict[str, np.ndarray]:
    """
    Run the full preprocessing stage for every configured color class.

    Returns a dict mapping color name -> cleaned binary mask.
    """
    hsv = bgr_to_hsv(frame)
    masks = {}
    for color_name, color_def in color_classes.items():
        raw_mask = build_color_mask(hsv, color_def)
        masks[color_name] = clean_mask(raw_mask, morph_cfg)
    return masks
