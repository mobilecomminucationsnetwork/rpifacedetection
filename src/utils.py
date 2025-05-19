# utils.py
"""
Utility functions and shared state for the Face Door Control project.
Includes:
- Global AUTHORIZED list of (name, vector)
- fetch_authorized_faces(): populate AUTHORIZED from API
- align_and_crop(): align face based on landmarks
"""

import numpy as np
import cv2
import requests
from config import FACE_DATA_URL

# Global list of known faces: list of tuples (name: str, vector: np.ndarray)
AUTHORIZED: list[tuple[str, np.ndarray]] = []

def fetch_authorized_faces() -> None:
    """
    Fetch authorized face embeddings from the server and populate AUTHORIZED.
    Expects JSON with 'results' list of entries having 'name' and 'vector_data'.
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
        print(vec)
        if name and isinstance(vec, list):
            try:
                arr = np.array(vec, dtype=np.float32)
                AUTHORIZED.append((name, arr))
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
