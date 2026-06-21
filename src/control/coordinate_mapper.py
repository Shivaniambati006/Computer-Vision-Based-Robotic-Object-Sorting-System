"""
src/control/coordinate_mapper.py

Translates pixel-space object centroids into physical robot-workspace
coordinates, using a calibrated linear or homography-based transform.
"""

from typing import Tuple

import numpy as np


class CoordinateMapper:
    """
    Maps 2D pixel coordinates (cx, cy) from the camera frame to 3D
    physical coordinates (x, y, z) in the robotic arm's workspace.
    """

    def __init__(self, mapping_cfg: dict):
        self.method = mapping_cfg.get("method", "linear")
        self.origin_pixel = tuple(mapping_cfg.get("origin_pixel", [0, 0]))
        self.scale_x = mapping_cfg.get("scale_x_mm_per_px", 1.0)
        self.scale_y = mapping_cfg.get("scale_y_mm_per_px", 1.0)
        self.z_pick_height = mapping_cfg.get("z_pick_height_mm", 0)
        self._homography = None  # populated if method == "homography"

    def set_homography(self, src_points, dst_points):
        """
        Compute a homography matrix from corresponding pixel/world
        point pairs (at least 4 points each), enabling more accurate
        mapping than the simple linear model — recommended for
        non-coplanar or angled camera mounts.
        """
        import cv2

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self._homography, _ = cv2.findHomography(src, dst)
        self.method = "homography"

    def pixel_to_world(self, centroid: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Convert a pixel-space centroid to a physical (x, y, z) coordinate
        in the robot's workspace frame.
        """
        cx, cy = centroid

        if self.method == "homography" and self._homography is not None:
            import cv2

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pt, self._homography)[0][0]
            return float(world_pt[0]), float(world_pt[1]), float(self.z_pick_height)

        # Default: linear mapping relative to a calibrated origin pixel.
        ox, oy = self.origin_pixel
        world_x = (cx - ox) * self.scale_x
        world_y = (cy - oy) * self.scale_y
        return float(world_x), float(world_y), float(self.z_pick_height)
