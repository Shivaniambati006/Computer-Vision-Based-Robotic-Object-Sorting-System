"""
main.py

Entry point for the Computer Vision-Based Robotic Object Sorting System.

Runs the full pipeline loop:
    Video Capture -> Preprocessing (HSV + masking + morphology)
    -> Detection (contours + localization) -> Sorting Logic
    -> Robotic Arm Dispatch

Usage:
    python main.py
    python main.py --debug          # enable visualization overlay
    python main.py --source 1       # use a different camera index
"""

import argparse
import os

import cv2

from src.vision.preprocessing import load_yaml, segment_all_colors
from src.vision.detection import localize_all
from src.control.coordinate_mapper import CoordinateMapper
from src.control.sorting_logic import ArmInterface, SortingEngine
from src.utils.visualization import (
    FPSCounter,
    draw_detections,
    draw_fps,
    draw_sorting_rate,
)
from src.utils.logger import setup_logger, PerformanceLogger


CONFIG_DIR = "config"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CV-based robotic object sorting pipeline."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug visualization overlay."
    )
    parser.add_argument(
        "--source", type=int, default=None, help="Override camera source index."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    color_cfg = load_yaml(os.path.join(CONFIG_DIR, "color_thresholds.yaml"))
    system_cfg = load_yaml(os.path.join(CONFIG_DIR, "system_config.yaml"))

    color_classes = color_cfg["color_classes"]
    min_contour_area = color_cfg.get("min_contour_area", 500)

    logger = setup_logger(
        log_dir=system_cfg["logging"]["log_dir"],
        log_level=system_cfg["logging"]["log_level"],
    )
    perf_logger = PerformanceLogger(
        logger, interval_sec=system_cfg["logging"]["performance_log_interval_sec"]
    )

    camera_source = (
        args.source if args.source is not None else system_cfg["camera"]["source"]
    )
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, system_cfg["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, system_cfg["camera"]["frame_height"])

    if not cap.isOpened():
        logger.error(f"Unable to open camera source: {camera_source}")
        return

    coordinate_mapper = CoordinateMapper(system_cfg["coordinate_mapping"])
    arm = ArmInterface(system_cfg["robot_arm"])
    sorting_engine = SortingEngine(color_classes, coordinate_mapper, arm)

    fps_counter = FPSCounter()
    debug_mode = args.debug or system_cfg["pipeline"]["debug_overlay_default"]

    logger.info("Starting sorting pipeline. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera. Stopping.")
                break

            # --- Vision Pipeline ---
            masks = segment_all_colors(
                frame, color_classes, system_cfg["morphology"]
            )
            detections = localize_all(masks, min_contour_area)

            # --- Sorting Logic / Robotic Dispatch ---
            sorting_engine.process_detections(detections)

            fps = fps_counter.tick()
            rate = sorting_engine.sorting_rate_per_minute()
            perf_logger.maybe_log(sorting_engine.sorted_count, rate, fps)

            # --- Debug Visualization ---
            if debug_mode:
                display_frame = frame.copy()
                draw_detections(display_frame, detections)
                draw_fps(display_frame, fps)
                draw_sorting_rate(display_frame, rate, sorting_engine.sorted_count)
                cv2.imshow("Robotic Sorting System - Debug View", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Still allow quitting via 'q' if a window happens to be focused,
                # and avoid a tight unthrottled loop.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        arm.return_home()
        cap.release()
        cv2.destroyAllWindows()
        logger.info(
            f"Session ended. Total sorted: {sorting_engine.sorted_count} "
            f"| Final rate: {sorting_engine.sorting_rate_per_minute():.2f}/min"
        )


if __name__ == "__main__":
    main()
