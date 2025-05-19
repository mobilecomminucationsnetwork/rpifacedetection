# camera.py
"""
CameraThread encapsulates Picamera2 capture in a background thread,
providing the latest frame on demand.
"""
import threading
import time
from picamera2 import Picamera2
from config import CAMERA_FORMAT, CAMERA_SIZE

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        # Initialize Picamera2 with preview configuration
        self.picam2 = Picamera2()
        self.cfg = self.picam2.create_preview_configuration(
            main={"format": CAMERA_FORMAT, "size": CAMERA_SIZE}
        )
        self.picam2.configure(self.cfg)

        # Shared frame buffer and lock
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        """
        Continuously capture frames and store the latest one.
        """
        self.picam2.start()
        try:
            while True:
                img = self.picam2.capture_array()
                with self.lock:
                    self.frame = img
                # throttle capture rate slightly
                time.sleep(0.01)
        finally:
            self.picam2.stop()

    def get_frame(self):
        """
        Return the most recent frame (or None if not yet available).
        """
        with self.lock:
            return self.frame
