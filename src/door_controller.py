# door_controller.py
"""
DoorController encapsulates servo initialization and door state management.
"""
import time
import servo
from config import OPEN_ANGLE, CLOSED_ANGLE, HOLD_TIME
from notifier import notify_status

class DoorController:
    def __init__(self):
        """
        Initialize the servo hardware and ensure door starts in closed position.
        """
        # Initialize servo PWM and set to closed angle
        servo.init_servo()
        self.is_open = False
        self._move(CLOSED_ANGLE)
        notify_status("CLOSED")


    def _move(self, angle: float) -> None:
        """
        Move servo to given angle and hold briefly to ensure position.
        """
        servo._move_to(angle)  # assumes servo module provides internal move
        time.sleep(HOLD_TIME)

    def open(self) -> None:
        """
        Open the door (move servo to OPEN_ANGLE) if not already open.
        """
        if not self.is_open:
            self._move(OPEN_ANGLE)
            self.is_open = True
            notify_status("OPEN")

    def close(self) -> bool:
        """
        Close the door (move servo to CLOSED_ANGLE) if not already closed.
        Returns True if door was closed, False if already closed.
        """
        if self.is_open:
            self._move(CLOSED_ANGLE)
            self.is_open = False
            notify_status("CLOSED")
            return True
        return False
