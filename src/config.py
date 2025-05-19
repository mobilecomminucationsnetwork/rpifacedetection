# config.py
"""
All configuration constants for the Face Door Control project.
"""

# ─── SERVER & API ─────────────────────────────────────────────────────
SERVER_URL       = "http://161.35.195.142:8000"
FACE_DATA_URL    = f"{SERVER_URL}/api/face-vectors/"
ANON_STORE_URL   = f"{SERVER_URL}/api/anonymous-face-vectors/"
DOOR_ID          = "e43b48ac-6cce-430e-a119-5c5ff5d62967"
DOOR_STATUS_URL  = f"{SERVER_URL}/api/doors/{DOOR_ID}/set-status/"
WS_URL           = SERVER_URL.replace("http://", "ws://") + f"/ws/doors/{DOOR_ID}/"

# ─── FACE RECOGNITION ─────────────────────────────────────────────────
# Thresholds for matching embeddings
THRESHOLD        = 0.6    # cosine similarity for known faces
ANON_MATCH_THRES = 0.7    # similarity to group same unknown

# Timing (in seconds)
MIN_CLOSE_DELAY        = 10.0   # auto-close door after last seen
UNKNOWN_DELAY          = 3.0    # wait before notifying unknown
UNKNOWN_COOLDOWN       = 30.0   # seconds between unknown notifications
STALE_DURATION         = 10.0   # retain unknowns this long in history
RECOGNIZED_DELAY       = 3.0    # must remain detected this long to open

# ─── MODELS ───────────────────────────────────────────────────────────\# SCRFD for face detection
FACE_DET_MODEL      = "scrfd_2.5g--640x640_quant_hailort_hailo8l_1"
# ArcFace for face recognition
FACE_REC_MODEL      = "arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1"
INFERENCE_HOST      = "@local"
ZOO_URL             = "models"

# ─── SERVO (DOOR) ─────────────────────────────────────────────────────
# BCM pin for servo signal
SERVO_PIN        = 18
# PWM parameters
PWM_FREQUENCY    = 50      # Hz
# Angles for door positions
OPEN_ANGLE       = 0     # degrees (servo open position)
CLOSED_ANGLE     = 140      # degrees (servo closed position)
# How long to hold the pulse at each move
HOLD_TIME        = 1    # seconds

# ─── CAMERA ────────────────────────────────────────────────────────────
CAMERA_FORMAT    = "RGB888"
CAMERA_SIZE      = (640, 640)
STREAM_SIZE      = (640, 640)  # MJPEG/WS stream resolution

# ─── LOGGING & MISC ───────────────────────────────────────────────────
LOG_LEVEL        = "INFO"   # DEBUG, INFO, WARN, ERROR


# Ultrasonic sensor (HC-SR04)
ULTRASONIC_TRIG_PIN        = 23   # BCM pin for TRIG
ULTRASONIC_ECHO_PIN        = 24   # BCM pin for ECHO
ULTRASONIC_THRESHOLD_CM    = 10.0  # distance under which we trigger
ULTRASONIC_HOLD_TIME_S     = 7.0  # must stay under threshold this long
ULTRASONIC_MEASURE_INTERVAL_S = 0.1  # how often to measure (seconds)

UNKNOWN_SIM_THRESHOLD = 0.1
# End of config.py

