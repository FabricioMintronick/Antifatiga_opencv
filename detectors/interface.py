# detectors/interface.py
import numpy as np

class IFaceDetector:
    def detect(self, frame_bgr) -> np.ndarray:
        """
        Recibe un frame BGR (numpy array) y devuelve:
        np.ndarray (N x 3) de landmarks normalizados (0-1)
        o None si no hay rostro.
        """
        raise NotImplementedError("Método detect() no implementado")
