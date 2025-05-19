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
UNKNOWN_SIM_THRESHOLD = 0.1

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
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            dets = self.face_det(frame).results
            if not dets:
                time.sleep(0.01)
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
                        for name, vec in AUTHORIZED:
                            vec_norm = vec / np.linalg.norm(vec)
                            sim = float(np.dot(e, vec_norm))
                            # print(sim)
                            if sim > best_sim:
                                best_sim, best_name = sim, name

                        # only emit events for clear cases:
                        if best_sim >= THRESHOLD:
                            self.queue.put(("recognized", best_name, frame))
                        elif best_sim <= UNKNOWN_SIM_THRESHOLD:
                            self.queue.put(("unknown", e.tolist(), frame))
                        # else: too ambiguous—ignore


            time.sleep(0.01)
            

            


