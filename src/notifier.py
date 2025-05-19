# notifier.py
"""
Notifier module handles all outbound HTTP notifications:
- Door status updates
- Unknown-face image uploads
"""
import base64
import cv2
import requests
import datetime
from typing import Optional
from config import ANON_STORE_URL, DOOR_STATUS_URL


def notify_status(status: str) -> None:
    """
    Send a door status update to the server.

    :param status: "OPEN" or "CLOSED"
    """
    payload = {"status": status}
    try:
        resp = requests.post(DOOR_STATUS_URL, json=payload, timeout=5)
        resp.raise_for_status()
        print(f"[NOTIFIER] Door status '{status}' sent ({resp.status_code})")
    except requests.RequestException as exc:
        print(f"[NOTIFIER] Failed to send door status '{status}': {exc}")


def notify_unknown_face(frame: Optional[cv2.UMat], embedding: Optional[list]) -> None:
    """
    Upload an unknown person's face image and embedding to the server.

    :param frame: BGR or RGB image array; if None, function returns immediately
    :param embedding: list of floats representing the face vector
    """
    if frame is None or embedding is None:
        return

    # Encode image to JPEG
    success, buf = cv2.imencode('.jpg', frame)
    if not success:
        print("[NOTIFIER] Failed to encode frame to JPEG")
        return

    # Base64 payload
    b64jpg = base64.b64encode(buf).decode('utf-8')
    payload = {
        'name': 'Unknown Person',
        'vector_data': embedding,
        'face_image_base64': f"data:image/jpeg;base64,{b64jpg}",
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    try:
        resp = requests.post(ANON_STORE_URL, json=payload, timeout=5)
        resp.raise_for_status()
        print(f"[NOTIFIER] Unknown face uploaded ({resp.status_code})")
    except requests.RequestException as exc:
        print(f"[NOTIFIER] Failed to upload unknown face: {exc}")
