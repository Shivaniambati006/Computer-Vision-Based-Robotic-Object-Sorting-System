"""
src/control/sorting_logic.py

Decision-making layer that converts localized, classified vision
detections into robotic arm pick-and-place commands.

This module is hardware-agnostic: the ArmInterface abstraction supports
a "simulation" backend (default, logs/prints commands) as well as
placeholders for serial and TCP-connected physical arms.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from src.control.coordinate_mapper import CoordinateMapper
from src.vision.detection import DetectedObject


@dataclass
class SortCommand:
    """A single pick-and-place command dispatched to the robotic arm."""

    color: str
    pick_coordinate: tuple   # (x, y, z) in robot workspace
    place_coordinate: tuple  # (x, y, z) target bin location
    bin_id: str
    timestamp: float


class ArmInterface:
    """
    Hardware abstraction layer for the robotic arm.

    In "simulation" mode, commands are logged rather than sent over a
    physical link — useful for pipeline validation without hardware
    attached. Swap in serial/TCP send logic for physical deployment.
    """

    def __init__(self, arm_cfg: dict):
        self.interface_type = arm_cfg.get("interface", "simulation")
        self.home_position = tuple(arm_cfg.get("home_position", [0, 0, 100]))
        self.approach_height = arm_cfg.get("approach_height_mm", 50)
        self._connection = None

        if self.interface_type == "serial":
            self._init_serial(arm_cfg)
        elif self.interface_type == "tcp":
            self._init_tcp(arm_cfg)
        # "simulation" requires no connection setup

    def _init_serial(self, arm_cfg: dict):
        try:
            import serial

            self._connection = serial.Serial(
                arm_cfg["serial_port"], arm_cfg["baud_rate"], timeout=1
            )
        except Exception as e:
            print(f"[WARN] Serial connection failed, falling back to simulation: {e}")
            self.interface_type = "simulation"

    def _init_tcp(self, arm_cfg: dict):
        try:
            import socket

            self._connection = socket.create_connection(
                (arm_cfg["tcp_host"], arm_cfg["tcp_port"]), timeout=2
            )
        except Exception as e:
            print(f"[WARN] TCP connection failed, falling back to simulation: {e}")
            self.interface_type = "simulation"

    def execute_pick_and_place(self, command: SortCommand):
        """Dispatch a pick-and-place command to the configured backend."""
        if self.interface_type == "simulation":
            print(
                f"[SIM] PICK {command.color.upper()} at {command.pick_coordinate} "
                f"-> PLACE in {command.bin_id} at {command.place_coordinate}"
            )
        elif self.interface_type == "serial" and self._connection:
            payload = self._encode_command(command)
            self._connection.write(payload)
        elif self.interface_type == "tcp" and self._connection:
            payload = self._encode_command(command)
            self._connection.sendall(payload)
        else:
            print("[WARN] No active arm connection; command not sent.")

    @staticmethod
    def _encode_command(command: SortCommand) -> bytes:
        """Serialize a command into a simple wire protocol (placeholder)."""
        msg = (
            f"PICK,{command.pick_coordinate[0]:.2f},{command.pick_coordinate[1]:.2f},"
            f"{command.pick_coordinate[2]:.2f};PLACE,{command.place_coordinate[0]:.2f},"
            f"{command.place_coordinate[1]:.2f},{command.place_coordinate[2]:.2f}\n"
        )
        return msg.encode("utf-8")

    def return_home(self):
        """Send the arm back to its idle/home position."""
        if self.interface_type == "simulation":
            print(f"[SIM] Returning to home position {self.home_position}")


class SortingEngine:
    """
    Core decision-making engine: maps detected objects to sort
    commands and dispatches them to the robotic arm.
    """

    def __init__(
        self,
        color_classes: dict,
        coordinate_mapper: CoordinateMapper,
        arm: ArmInterface,
    ):
        self.color_classes = color_classes
        self.coordinate_mapper = coordinate_mapper
        self.arm = arm
        self.sorted_count = 0
        self._session_start = time.time()

    def process_detections(
        self, detections: List[DetectedObject]
    ) -> List[SortCommand]:
        """
        Convert a frame's detected objects into dispatched sort commands.
        Objects are processed in descending order of contour area,
        prioritizing the most confidently detected object first.
        """
        commands: List[SortCommand] = []
        detections_sorted = sorted(detections, key=lambda d: d.area, reverse=True)

        for obj in detections_sorted:
            command = self._build_command(obj)
            if command is None:
                continue
            self.arm.execute_pick_and_place(command)
            commands.append(command)
            self.sorted_count += 1

        return commands

    def _build_command(self, obj: DetectedObject) -> Optional[SortCommand]:
        """Build a SortCommand for a single detected object, if its
        color class is recognized in the configuration."""
        color_def = self.color_classes.get(obj.color)
        if color_def is None:
            return None

        pick_coord = self.coordinate_mapper.pixel_to_world(obj.centroid)
        place_coord = tuple(color_def["bin_coordinates"])
        bin_id = color_def.get("bin_id", "unknown")

        return SortCommand(
            color=obj.color,
            pick_coordinate=pick_coord,
            place_coordinate=place_coord,
            bin_id=bin_id,
            timestamp=time.time(),
        )

    def sorting_rate_per_minute(self) -> float:
        """Compute the rolling sorting throughput rate (objects/minute)."""
        elapsed_min = max((time.time() - self._session_start) / 60.0, 1e-6)
        return self.sorted_count / elapsed_min
