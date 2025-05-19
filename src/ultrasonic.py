# ultrasonic.py
import threading
import time
import RPi.GPIO as GPIO

from config import (
    ULTRASONIC_TRIG_PIN,
    ULTRASONIC_ECHO_PIN,
    ULTRASONIC_THRESHOLD_CM,
    ULTRASONIC_HOLD_TIME_S,
    ULTRASONIC_MEASURE_INTERVAL_S
)

# how long (s) we’re willing to wait for an echo change
ECHO_TIMEOUT = 0.02  

class UltrasonicThread(threading.Thread):
    def __init__(self, command_queue):
        super().__init__(daemon=True)
        self.queue = command_queue
        self.last_distance = None  # Track last valid distance
        self._sensor_below_since = None
        self._should_auto_close = False
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([ULTRASONIC_TRIG_PIN], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup([ULTRASONIC_ECHO_PIN], GPIO.IN)

    def measure_distance(self) -> float | None:
        # trigger pulse
        GPIO.output(ULTRASONIC_TRIG_PIN, True)
        time.sleep(10e-6)
        GPIO.output(ULTRASONIC_TRIG_PIN, False)

        # wait for echo to go high, with timeout
        start_time = time.time()
        while GPIO.input(ULTRASONIC_ECHO_PIN) == 0:
            if time.time() - start_time > ECHO_TIMEOUT:
                return None
        t0 = time.time()

        # wait for echo to go low, with timeout
        while GPIO.input(ULTRASONIC_ECHO_PIN) == 1:
            if time.time() - t0 > ECHO_TIMEOUT:
                return None
        t1 = time.time()

        duration = t1 - t0
        return (duration * 34300) / 2  # cm

    def should_auto_close(self):
        return self._should_auto_close

    def reset_auto_close(self):
        self._sensor_below_since = None
        self._should_auto_close = False

    def run(self):
        while True:
            dist = self.measure_distance()
            self.last_distance = dist
            now = time.time()
            if dist is not None and dist <= ULTRASONIC_THRESHOLD_CM:
                if self._sensor_below_since is None:
                    self._sensor_below_since = now
                elif (now - self._sensor_below_since) >= ULTRASONIC_HOLD_TIME_S:
                    self._should_auto_close = True
            else:
                self._sensor_below_since = None
                self._should_auto_close = False
            time.sleep(ULTRASONIC_MEASURE_INTERVAL_S)
