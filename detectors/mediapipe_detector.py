# detectors/mediapipe_detector.py
import mediapipe as mp
import numpy as np
import cv2, time
from detectors.interface import IFaceDetector

class MediapipeDetector(IFaceDetector):
    def __init__(self, model_path: str):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )
        self.detector = FaceLandmarker.create_from_options(self.options)

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        return lm
