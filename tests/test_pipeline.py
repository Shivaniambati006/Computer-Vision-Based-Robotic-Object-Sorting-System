"""
tests/test_pipeline.py

Unit tests covering the core vision pipeline stages: HSV conversion,
color masking, morphological cleaning, contour detection, and
centroid/bounding-box localization.

Run with:
    pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest

from src.vision.preprocessing import bgr_to_hsv, build_color_mask, clean_mask
from src.vision.detection import (
    compute_centroid,
    find_contours,
    localize_objects,
)


@pytest.fixture
def synthetic_red_square_frame():
    """
    Build a synthetic BGR frame containing a solid red square on a
    black background, used to validate the pipeline without requiring
    a live camera.
    """
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # OpenCV uses BGR ordering -> pure red is (0, 0, 255)
    frame[50:150, 50:150] = (0, 0, 255)
    return frame


@pytest.fixture
def red_color_def():
    return {
        "lower1": [0, 120, 70],
        "upper1": [10, 255, 255],
        "lower2": [170, 120, 70],
        "upper2": [179, 255, 255],
    }


@pytest.fixture
def morph_cfg():
    return {
        "erosion_kernel": [3, 3],
        "erosion_iterations": 1,
        "dilation_kernel": [3, 3],
        "dilation_iterations": 1,
    }


def test_bgr_to_hsv_shape_preserved(synthetic_red_square_frame):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    assert hsv.shape == synthetic_red_square_frame.shape


def test_color_mask_isolates_red_region(synthetic_red_square_frame, red_color_def):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    mask = build_color_mask(hsv, red_color_def)

    # The masked region should be non-zero where the red square is.
    assert mask[100, 100] == 255
    # And zero in the black background area.
    assert mask[10, 10] == 0


def test_clean_mask_preserves_object_presence(
    synthetic_red_square_frame, red_color_def, morph_cfg
):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    raw_mask = build_color_mask(hsv, red_color_def)
    cleaned = clean_mask(raw_mask, morph_cfg)

    assert cleaned.sum() > 0  # object should still be present after cleaning


def test_find_contours_detects_one_object(
    synthetic_red_square_frame, red_color_def, morph_cfg
):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    mask = clean_mask(build_color_mask(hsv, red_color_def), morph_cfg)
    contours = find_contours(mask)

    assert len(contours) == 1


def test_compute_centroid_matches_expected_center(
    synthetic_red_square_frame, red_color_def, morph_cfg
):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    mask = clean_mask(build_color_mask(hsv, red_color_def), morph_cfg)
    contours = find_contours(mask)
    cx, cy = compute_centroid(contours[0])

    # The square spans (50,50) to (150,150) -> expected center ~ (100, 100)
    assert 90 <= cx <= 110
    assert 90 <= cy <= 110


def test_localize_objects_returns_correct_color_label(
    synthetic_red_square_frame, red_color_def, morph_cfg
):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    mask = clean_mask(build_color_mask(hsv, red_color_def), morph_cfg)
    detected = localize_objects(mask, "red", min_contour_area=500)

    assert len(detected) == 1
    assert detected[0].color == "red"
    assert detected[0].area > 0


def test_localize_objects_filters_by_min_area(
    synthetic_red_square_frame, red_color_def, morph_cfg
):
    hsv = bgr_to_hsv(synthetic_red_square_frame)
    mask = clean_mask(build_color_mask(hsv, red_color_def), morph_cfg)

    # Setting an unreasonably high min area should filter the object out.
    detected = localize_objects(mask, "red", min_contour_area=999999)
    assert len(detected) == 0
