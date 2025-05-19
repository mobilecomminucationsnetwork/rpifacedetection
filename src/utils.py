# utils.py
"""
Utility functions and shared state for the Face Door Control project.
Includes:
- Global AUTHORIZED list of (name, vector, image)
- fetch_authorized_faces(): populate AUTHORIZED from API
- align_and_crop(): align face based on landmarks
- encode_image_to_base64(): encode images to base64 for sending
"""

import numpy as np
import cv2
import requests
import base64
from config import FACE_DATA_URL

# Global list of known faces: list of tuples (name: str, vector: np.ndarray, image: np.ndarray)
AUTHORIZED: list[tuple[str, np.ndarray, np.ndarray]] = []

def fetch_authorized_faces() -> None:
    """
    Fetch authorized face embeddings from the server and populate AUTHORIZED.
    Expects JSON with 'results' list of entries having 'name', 'vector_data', and 'face_image_base64'.
    """
    global AUTHORIZED
    AUTHORIZED.clear()
    try:
        resp = requests.get(FACE_DATA_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[utils] Error fetching authorized faces: {e}")
        return

    entries = data.get("results") if isinstance(data, dict) else data
    for entry in entries or []:
        name = entry.get("name")
        vec  = entry.get("vector_data", entry.get("face_vector", []))
        face_b64 = entry.get("face_image_base64")
        img = None
        if face_b64:
            try:
                if face_b64.startswith("data:image"):
                    face_b64 = face_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(face_b64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as ex:
                print(f"[utils] Skipping invalid face image for {name}: {ex}")
        if name and isinstance(vec, list):
            try:
                arr = np.array(vec, dtype=np.float32)
                AUTHORIZED.append((name, arr, img))  # store image as well
            except Exception as ex:
                print(f"[utils] Skipping invalid vector for {name}: {ex}")
    print(f"[utils] Loaded {len(AUTHORIZED)} authorized faces")

def align_and_crop(img: np.ndarray, landmarks: list[list[float]], size: int = 112) -> np.ndarray:
    """
    Align and crop a face from img using 5-point landmarks.

    :param img: source image (H x W x C)
    :param landmarks: list of 5 [x,y] points
    :param size: output square size (pixels)
    :return: aligned, cropped image of shape (size, size, C)
    """
    # Reference points for a 112x112 face
    ref = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32) * (size / 112.0)

    pts = np.array(landmarks, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts, ref)
    aligned = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
    return aligned

def encode_image_to_base64(img: np.ndarray) -> str:
    """
    Encode a BGR or RGB image to base64 JPEG string with data URI prefix.
    """
    success, buf = cv2.imencode('.jpg', img)
    if not success:
        return ""
    b64jpg = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64jpg}"
