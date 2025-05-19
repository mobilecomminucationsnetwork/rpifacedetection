# recognizer.py
"""
RecognizerThread continuously reads frames from the camera, performs face detection
and recognition, then pushes events to a shared queue:
- ('recognized', name, frame)
- ('unknown', embedding, frame)
"""
import threading
import time
import numpy as np
import degirum as dg

from config import FACE_DET_MODEL, FACE_REC_MODEL, INFERENCE_HOST, ZOO_URL, THRESHOLD
from utils import align_and_crop, fetch_authorized_faces, AUTHORIZED

# below this sim or lower, we treat as "unknown"
UNKNOWN_SIM_THRESHOLD = 0.4  # was 0.1, now more reasonable

class RecognizerThread(threading.Thread):
    def __init__(self, camera, event_queue):
        super().__init__(daemon=True)
        self.camera = camera
        self.queue  = event_queue

        # Load models
        self.face_det = dg.load_model(
            model_name             = FACE_DET_MODEL,
            inference_host_address = INFERENCE_HOST,
            zoo_url                = ZOO_URL
        )
        self.face_rec = dg.load_model(
            model_name             = FACE_REC_MODEL,
            inference_host_address = INFERENCE_HOST,
            zoo_url                = ZOO_URL
        )

        # Populate AUTHORIZED once
        fetch_authorized_faces()

    def run(self):
        last_unknown_emb = None
        unknown_start_time = None
        UNKNOWN_HOLD_TIME = 3.0
        UNKNOWN_DIFF_THRESHOLD = 0.8  # Only reset timer if embedding is very different
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                unknown_start_time = None
                last_unknown_emb = None
                continue

            dets = self.face_det(frame).results
            if not dets:
                time.sleep(0.01)
                unknown_start_time = None
                last_unknown_emb = None
                continue

            det = dets[0]
            lms = det.get("landmarks", [])
            bbox = det.get("bbox", det.get("box", []))

            if len(lms) == 5 and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                    pts  = [lm_["landmark"] for lm_ in lms]
                    face = align_and_crop(frame, pts)

                    res = self.face_rec(face).results
                    emb = res[0].get("data", [[None]])[0]
                    if emb is not None:
                        e = np.array(emb, dtype=np.float32)
                        e /= np.linalg.norm(e)

                        # find best match
                        best_sim, best_name = 0.0, None
                        for name, vec, _ in AUTHORIZED:
                            vec_norm = vec / np.linalg.norm(vec)
                            sim = float(np.dot(e, vec_norm))
                            if sim > best_sim:
                                best_sim, best_name = sim, name

                        # only emit events for clear cases:
                        if best_sim >= THRESHOLD:
                            self.queue.put(("recognized", best_name, frame))
                            unknown_start_time = None
                            last_unknown_emb = None
                        elif best_sim <= UNKNOWN_SIM_THRESHOLD:
                            # If new unknown or embedding changed significantly, reset timer
                            if last_unknown_emb is None or np.linalg.norm(e - last_unknown_emb) > UNKNOWN_DIFF_THRESHOLD:
                                unknown_start_time = time.time()
                                last_unknown_emb = e
                                print("[recognizer] new unknown detected, timer started")
                            else:
                                # If same unknown, check hold time
                                if unknown_start_time and (time.time() - unknown_start_time) >= UNKNOWN_HOLD_TIME:
                                    self.queue.put(("unknown", e.tolist(), frame))
                                    print("[recognizer] unknown in sight > 3s, event sent")
                                    unknown_start_time = None  # Only send once per sighting
                                    last_unknown_emb = None
                        else:
                            unknown_start_time = None
                            last_unknown_emb = None
            else:
                unknown_start_time = None
                last_unknown_emb = None
            time.sleep(0.01)





