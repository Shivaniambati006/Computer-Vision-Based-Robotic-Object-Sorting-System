"""
src/vision/calibration.py

Interactive HSV threshold calibration utility.

Run this script to live-tune Hue/Saturation/Value lower and upper bounds
against your camera feed and lighting conditions. Once the mask preview
cleanly isolates only the target object, copy the resulting range into
config/color_thresholds.yaml.

Usage:
    python src/vision/calibration.py
"""

import cv2
import numpy as np


WINDOW_TRACKBARS = "HSV Calibration"
WINDOW_MASK = "Mask Preview"
WINDOW_ORIGINAL = "Original Feed"


def _nothing(_):
    """Dummy callback required by cv2.createTrackbar."""
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_TRACKBARS)
    cv2.createTrackbar("H Min", WINDOW_TRACKBARS, 0, 179, _nothing)
    cv2.createTrackbar("H Max", WINDOW_TRACKBARS, 179, 179, _nothing)
    cv2.createTrackbar("S Min", WINDOW_TRACKBARS, 0, 255, _nothing)
    cv2.createTrackbar("S Max", WINDOW_TRACKBARS, 255, 255, _nothing)
    cv2.createTrackbar("V Min", WINDOW_TRACKBARS, 0, 255, _nothing)
    cv2.createTrackbar("V Max", WINDOW_TRACKBARS, 255, 255, _nothing)


def read_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", WINDOW_TRACKBARS)
    h_max = cv2.getTrackbarPos("H Max", WINDOW_TRACKBARS)
    s_min = cv2.getTrackbarPos("S Min", WINDOW_TRACKBARS)
    s_max = cv2.getTrackbarPos("S Max", WINDOW_TRACKBARS)
    v_min = cv2.getTrackbarPos("V Min", WINDOW_TRACKBARS)
    v_max = cv2.getTrackbarPos("V Max", WINDOW_TRACKBARS)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def run_calibration(camera_source=0):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {camera_source}")

    create_trackbars()
    print("[INFO] Calibration running. Press 'q' to quit, 's' to print current range.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = read_trackbar_values()
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow(WINDOW_ORIGINAL, frame)
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow("Segmented Result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"[CALIBRATED RANGE] lower1: {list(lower)}  upper1: {list(upper)}")
            print("Copy these values into config/color_thresholds.yaml")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
