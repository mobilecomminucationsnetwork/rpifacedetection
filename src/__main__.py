#!/usr/bin/env python3
import queue
import time
import sys

from config import (
    UNKNOWN_COOLDOWN,
    ULTRASONIC_THRESHOLD_CM,
    ULTRASONIC_HOLD_TIME_S,
)
from camera import CameraThread
from recognizer import RecognizerThread
from ws_client import WSClientThread
from ultrasonic import UltrasonicThread
from door_controller import DoorController
from notifier import notify_status, notify_unknown_face

def main():
    # Queues for WS/ultrasonic commands and recognition events
    cmd_q = queue.Queue()
    evt_q = queue.Queue()

    # Initialize threads and controller
    cam_thread        = CameraThread()
    recog_thread      = RecognizerThread(cam_thread, evt_q)
    ws_thread         = WSClientThread(cmd_q)
    ultrasonic_thread = UltrasonicThread(cmd_q)
    door_ctrl         = DoorController()

    # Start all threads
    cam_thread.start()
    recog_thread.start()
    ws_thread.start()
    ultrasonic_thread.start()

    last_unknown       = 0.0
    sensor_below_since = None

    try:
        while True:
            now = time.time()

            # read most recent distance measurement
            dist = getattr(ultrasonic_thread, "last_distance", None)

            # 2) If object has been under threshold for ULTRASONIC_HOLD_TIME_S, auto-close
            if dist is not None and dist <= ULTRASONIC_THRESHOLD_CM:
                if sensor_below_since is None:
                    sensor_below_since = now
                elif (now - sensor_below_since) >= ULTRASONIC_HOLD_TIME_S:
                    closed = door_ctrl.close()

                    if closed:
                        notify_status("CLOSED")
                        ws_thread.send_closed_status()
                    sensor_below_since = None
            else:
                sensor_below_since = None

            # 1) Handle OPEN/CLOSED commands from WS (manual), but only allow CLOSE if dist > threshold is ignored
            while not cmd_q.empty():
                cmd = cmd_q.get()
                if cmd == "OPEN":
                    if door_ctrl.open():
                        notify_status("OPEN")
                        ws_thread.send_opened_status()
                elif cmd == "CLOSED":
                    # Only ignore CLOSE if distance is greater than threshold
                    if dist is not None and dist > ULTRASONIC_THRESHOLD_CM:
                        print(f"[MAIN] Ignored CLOSE command (distance={dist})")
                        ws_thread.send_closed_status()  # Send CLOSED via websocket if ignored
                    else:
                        if door_ctrl.close():
                            notify_status("CLOSED")
                            ws_thread.send_closed_status()

            # 3) Handle recognition events (immediate open on recognized)
            while not evt_q.empty():
                evt = evt_q.get()
                if evt[0] == "recognized":
                    if door_ctrl.open():
                        notify_status("OPEN")
                elif evt[0] == "unknown":
                    _, emb, frame = evt
                    if now - last_unknown >= UNKNOWN_COOLDOWN:
                        notify_unknown_face(frame, emb)
                        last_unknown = now

            # no inactivity auto-close; sensor logic covers closing

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting…")
        sys.exit(0)

if __name__ == '__main__':
    main()
