import mediapipe as mp
import numpy as np
import cv2, time
from detectors.interface import IFaceDetector


class MediapipeDetector(IFaceDetector):
    def __init__(self, model_path: str, delegate: str = "CPU"):
        """
        Inicializa el detector facial MediaPipe FaceLandmarker.
        delegate: "CPU", "GPU" o "NPU" según hardware disponible.
        """
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # -------------------------------------------------------------
        #  Seleccionar delegado (aceleración de hardware)
        # -------------------------------------------------------------
        if delegate.upper() == "GPU":
            selected_delegate = BaseOptions.Delegate.GPU
            print("[INFO] Usando delegado GPU para MediaPipe.")
        elif delegate.upper() == "NPU":
            selected_delegate = BaseOptions.Delegate.NNAPI  # Android / Owasys compatible
            print("[INFO] Usando delegado NPU/NNAPI para MediaPipe.")
        else:
            selected_delegate = BaseOptions.Delegate.CPU
            print("[INFO] Usando delegado CPU (modo estándar).")

        base_opts = BaseOptions(
            model_asset_path=model_path,
            delegate=selected_delegate
        )

        # Configuración general
        self.options = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        # Crear el detector
        try:
            self.detector = FaceLandmarker.create_from_options(self.options)
        except Exception as e:
            print(f"[ERROR] No se pudo inicializar FaceLandmarker: {e}")
            self.detector = None


    # -------------------------------------------------------------
    #  Método de detección principal
    # -------------------------------------------------------------
    def detect(self, frame_bgr):
        """
        Procesa un frame BGR (OpenCV) y devuelve landmarks (68 puntos) o None.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None

        lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        return lm
