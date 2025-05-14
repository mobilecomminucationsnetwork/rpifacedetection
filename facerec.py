#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import degirum as dg

def align_and_crop(img: np.ndarray, landmarks: list[list[float]], size: int = 112) -> np.ndarray:
    ref = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    pts = np.array(landmarks, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts, ref * (size / 112.0))
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)

def main():
    # Load models
    face_det = dg.load_model(
        model_name="scrfd_2.5g--640x640_quant_hailort_hailo8l_1",
        inference_host_address="@local",
        zoo_url="degirum/models_hailort"
    )
    face_rec = dg.load_model(
        model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1",
        inference_host_address="@local",
        zoo_url="degirum/models_hailort"
    )

    # Load your reference embedding
    ref_emb = np.load("user_embedding.npy")  # shape (128,)

    # Camera setup
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(cfg)
    picam2.start()
    cv2.namedWindow("Face Recog + ID", cv2.WINDOW_AUTOSIZE)

    threshold = 0.5  # cosine similarity threshold

    try:
        while True:
            frame = picam2.capture_array()
            dets = face_det(frame).results

            for face in dets:
                x1, y1, x2, y2 = map(int, face["bbox"])
                landmarks = [lm["landmark"] for lm in face["landmarks"]]

                # Align & embed
                crop = align_and_crop(frame, landmarks, size=112)
                emb = face_rec(crop).results[0]["data"][0]

                # Cosine similarity
                sim = np.dot(emb, ref_emb) / (np.linalg.norm(emb)*np.linalg.norm(ref_emb))
                name = "Furkan" if sim > threshold else "Unknown"

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Face Recog + ID", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()