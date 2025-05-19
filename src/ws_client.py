# ws_client.py
import threading
import json
import websocket
import base64
import cv2
import numpy as np
import requests
from utils import AUTHORIZED, align_and_crop  # <-- Import align_and_crop from utils
from config import WS_URL, FACE_DATA_URL

class WSClientThread(threading.Thread):
    def __init__(self, command_queue):
        super().__init__(daemon=True)
        self.queue = command_queue
        self.ws = None  # Store ws instance for sending messages

    def send_closed_status(self):
        if self.ws:
            try:
                msg = json.dumps({"type": "door_status", "status": "CLOSED"})
                self.ws.send(msg)
                print("[ws_client] Sent CLOSED status via websocket")
            except Exception as e:
                print(f"[ws_client] Failed to send CLOSED status: {e}")

    def send_opened_status(self):
        if self.ws:
            try:
                msg = json.dumps({"type": "door_status", "status": "OPENED"})
                self.ws.send(msg)
                print("[ws_client] Sent OPENED status via websocket")
            except Exception as e:
                print(f"[ws_client] Failed to send OPENED status: {e}")

    def handle_face_recognition_request(self, data):
        b64img = data.get("face_image_base64")
        request_id = data.get("request_id")
        name = data.get("name", f"user_{request_id}")
        user_id = data.get("user_id", 1)  # fallback if not provided
        device_id = data.get("device_id", "cam01")
        if not b64img:
            print("[ws_client] No image data in request")
            return
        try:
            if b64img.startswith("data:image"):
                b64img = b64img.split(",", 1)[1]
            img_bytes = base64.b64decode(b64img)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ws_client] Failed to decode image: {e}")
            return
        # Import recognizer models
        from recognizer import RecognizerThread
        recog = RecognizerThread(None, None)
        dets = recog.face_det(img).results
        if not dets:
            print("[ws_client] No face detected in image")
            return
        det = dets[0]
        lms = det.get("landmarks", [])
        bbox = det.get("bbox", det.get("box", []))
        if len(lms) == 5 and len(bbox) == 4:
            pts = [lm_["landmark"] for lm_ in lms]
            face = align_and_crop(img, pts)  # <-- Use align_and_crop from utils
            res = recog.face_rec(face).results
            emb = res[0].get("data", [[None]])[0]
            if emb is not None:
                e = np.array(emb, dtype=np.float32)
                e /= np.linalg.norm(e)
                AUTHORIZED.append((name, e))
                payload = {
                    "name": name,
                    "user": user_id,
                    "vector_data": e.tolist(),
                    "metadata": {"source": "camera", "device_id": device_id, "request_id": request_id}
                }
                try:
                    resp = requests.post(FACE_DATA_URL, json=payload, timeout=5)
                    resp.raise_for_status()
                    print(f"[ws_client] Face vector uploaded ({resp.status_code})")
                except Exception as ex:
                    print(f"[ws_client] Failed to upload face vector: {ex}")
            else:
                print("[ws_client] No embedding extracted from face")
        else:
            print("[ws_client] Invalid face landmarks or bbox")

    def on_message(self, ws, message):
        """
        Expected message format:
          {"type": "door_status", "status": "OPEN", "timestamp": "..."}
        We enqueue the 'status' string for the main loop to handle.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"[ws_client] Ignoring non-JSON message: {message!r}")
            return

        if data.get("type") == "face_recognition_request":
            self.handle_face_recognition_request(data)
            return

        if data.get("type") != "door_status":
            return

        status = data.get("status", "").upper()
        ts     = data.get("timestamp", "")
        print(f"[ws_client] door_status={status} @ {ts}")

        if status in ("OPEN", "CLOSED"):
            self.queue.put(status)

    def on_error(self, ws, error):
        print(f"[ws_client] WebSocket error: {error}")

    def on_close(self, ws, close_status, close_msg):
        print(f"[ws_client] Connection closed: {close_status} - {close_msg}")

    def on_open(self, ws):
        print(f"[ws_client] Connected to {WS_URL}")
        self.ws = ws  # Save ws instance

    def run(self):
        websocket.enableTrace(False)
        ws_app = websocket.WebSocketApp(
            WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws_app.run_forever()
