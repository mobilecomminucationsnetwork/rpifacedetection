# servo.py
"""
Servo control module for door locking mechanism.
Provides initialization, movement, and cleanup of the servo via RPi.GPIO.
"""
import RPi.GPIO as GPIO
import time
import atexit
from config import SERVO_PIN, PWM_FREQUENCY, OPEN_ANGLE, CLOSED_ANGLE, HOLD_TIME

# Internal PWM handle
_pwm = None

# ─── Initialization ──────────────────────────────────────────────────────────
def init_servo(pin: int = SERVO_PIN) -> None:
    """
    Initialize GPIO and PWM for servo control.
    Must be called before open_door() or close_door().
    """
    global _pwm
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    _pwm = GPIO.PWM(pin, PWM_FREQUENCY)
    _pwm.start(0.0)

# ─── Movement Helpers ────────────────────────────────────────────────────────
def _angle_to_duty_cycle(angle: float) -> float:
    """
    Convert angle (0–180) to duty cycle (2–12%).
    """
    return (angle / 18.0) + 2.0


def _move_to(angle: float) -> None:
    """
    Move servo to specified angle and hold for HOLD_TIME, then zero pulses.
    """
    global _pwm
    if _pwm is None:
        init_servo()
    dc = _angle_to_duty_cycle(angle)
    try:
        _pwm.ChangeDutyCycle(dc)
        time.sleep(HOLD_TIME)
        _pwm.ChangeDutyCycle(0.0)
    except Exception as e:
        print(f"[servo] movement error: {e}")

# ─── Door Control API ───────────────────────────────────────────────────────
def open_door() -> None:
    """Swing servo to OPEN_ANGLE position."""
    _move_to(OPEN_ANGLE)


def close_door() -> None:
    """Swing servo to CLOSED_ANGLE position."""
    _move_to(CLOSED_ANGLE)

# ─── Cleanup ─────────────────────────────────────────────────────────────────
def _cleanup() -> None:
    """
    Stop PWM and clean up GPIO on exit.
    """
    global _pwm
    try:
        if _pwm is not None:
            _pwm.stop()
    except Exception:
        pass
    finally:
        GPIO.cleanup()

atexit.register(_cleanup)
