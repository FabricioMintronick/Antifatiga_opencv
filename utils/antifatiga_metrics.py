import numpy as np

# =========================
# FUNCIONES DE GEOMETRÍA FACIAL
# =========================

def eye_aspect_ratio(eye):
    """Cálculo del EAR (Eye Aspect Ratio).
       Mide cuán abierto está el ojo.
       Valores bajos → ojo cerrado."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C + 1e-6)

def compute_EAR(lm_px):
    """Calcula EAR promedio entre ojo izquierdo y derecho."""
    left  = lm_px[[33,160,158,133,153,144], :2]
    right = lm_px[[362,385,387,263,373,380], :2]
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

def compute_MAR(lm_px):
    """Calcula MAR (Mouth Aspect Ratio): apertura de la boca."""
    A = np.linalg.norm(lm_px[14,:2] - lm_px[13,:2])
    B = np.linalg.norm(lm_px[78,:2] - lm_px[308,:2])
    return A / (B + 1e-6)

def estimate_angles_simple(lm_px):
    """Calcula orientación básica de cabeza (Yaw, Pitch, Roll)."""
    leftEyeCorner  = lm_px[33,:2]
    rightEyeCorner = lm_px[263,:2]
    noseTip        = lm_px[1,:2]
    mouthCenter    = lm_px[[13,14],:2].mean(axis=0)

    d = rightEyeCorner - leftEyeCorner
    roll = -np.degrees(np.arctan2(d[1], d[0]))
    eyesMid = (leftEyeCorner + rightEyeCorner) / 2.0
    interOcular = np.linalg.norm(d)
    yaw = 90 * (noseTip[0] - eyesMid[0]) / (interOcular + 1e-6)
    faceHeight = abs(mouthCenter[1] - eyesMid[1])
    pitch = -90 * (noseTip[1] - eyesMid[1]) / (faceHeight + 1e-6)

    # límites para estabilidad
    yaw   = np.clip(yaw,  -60, 60)
    pitch = np.clip(pitch,-60, 60)
    roll  = np.clip(roll, -60, 60)
    return float(yaw), float(pitch), float(roll)

def ema(prev, x, alpha=0.3):
    """Media móvil exponencial para suavizar ruidos."""
    if prev is None or np.isnan(prev):
        return x
    return (1.0 - alpha)*prev + alpha*x
