# =======================
#  Sistema Antifatiga v2
# =======================
# Autor: Joan Pérez
# Fecha: Octubre 2025
# Descripción:
# Detecta fatiga y distracción mediante análisis facial
# usando MediaPipe FaceLandmarker. Compatible con Owa5x
# y PC. Incluye calibración, visualización y streaming JSON.

import argparse, cv2, time, numpy as np, json, datetime, os, sys
from detectors.mediapipe_detector import MediapipeDetector
from utils.antifatiga_metrics import compute_EAR, compute_MAR, estimate_angles_simple, ema

# ------------------------------------------------------------
# 🔄 Función de publicación en tiempo real (para MQTT/HTTP)
# ------------------------------------------------------------
def publish_event(evt: dict):
    print(json.dumps(evt, ensure_ascii=False), flush=True)

# ------------------------------------------------------------
# 🧠 Programa principal
# ------------------------------------------------------------
def main():
    # ========================
    # 1. PARÁMETROS GENERALES
    # ========================
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--backend', type=str, default='mediapipe',
                    choices=['mediapipe', 'tflite_cpu', 'tflite_npu'])
    ap.add_argument('--model', type=str, default='models/face_landmarker.task')

    # ---- Calibración y presets ----
    ap.add_argument('--calib_secs', type=float, default=5.0)
    ap.add_argument('--asian_preset', action='store_true')

    # ---- Umbrales base ----
    ap.add_argument('--yaw_thr', type=float, default=20.0)
    ap.add_argument('--pitch_thr', type=float, default=20.0)
    ap.add_argument('--mar_yawn', type=float, default=0.60)

    # ---- Configuración ocular ----
    ap.add_argument('--ear_close_pct', type=float, default=0.75)
    ap.add_argument('--blink_min_ms', type=int, default=120)

    # ---- Registro / visualización ----
    ap.add_argument('--jsonl_path', type=str, default='stream_antifatiga.jsonl')
    ap.add_argument('--show', action='store_true')

    args = ap.parse_args()

    # ================================
    # 2. PRESET PARA OJOS RASGADOS
    # ================================
    if args.asian_preset and args.ear_close_pct > 0.7:
        args.ear_close_pct = 0.70
        args.mar_yawn = max(args.mar_yawn, 0.65)

    # ===========================
    # 3. INICIALIZACIÓN GENERAL
    # ===========================
    print("[INFO] Inicializando detector...")
    detector = MediapipeDetector(args.model)
    print("[INFO] Detector MediaPipe cargado.")

    print("[INFO] Inicializando cámara...")
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("[INFO] Cámara abierta correctamente.")

    if args.show:
        cv2.namedWindow("Antifatiga", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Antifatiga", 640, 480)

    # ==================================
    # 4. CALIBRACIÓN (mirar al frente)
    # ==================================
    yaw_base = pitch_base = 0.0
    ear_base_acc = 0.0
    samples = 0
    t_end = time.time() + args.calib_secs

    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok: continue
        lm = detector.detect(frame)
        if lm is not None:
            h, w = frame.shape[:2]
            lm_px = lm.copy(); lm_px[:,0]*=w; lm_px[:,1]*=h
            yaw, pitch, _ = estimate_angles_simple(lm_px)
            ear = compute_EAR(lm_px)
            yaw_base+=yaw; pitch_base+=pitch; ear_base_acc+=ear; samples+=1
        if args.show:
            vis=frame.copy()
            secs_left=max(0,int(t_end-time.time())+1)
            cv2.putText(vis,f"CALIBRANDO {secs_left}s - Mire al frente",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),3)
            cv2.imshow("Antifatiga",cv2.resize(vis,(640,480)))
            if cv2.waitKey(1)&0xFF==27: break

    if samples>0:
        yaw_base/=samples; pitch_base/=samples; EAR_BASE=ear_base_acc/samples
    else:
        yaw_base=pitch_base=0.0; EAR_BASE=0.30

    print(f"[INFO] Calibración: yaw_base={yaw_base:.1f}, pitch_base={pitch_base:.1f}, EAR_base={EAR_BASE:.2f}")
    pitch_base += 20

    # ===============================
    # 5. UMBRALES Y VARIABLES DE ESTADO
    # ===============================
    ear_thresh_closed=EAR_BASE*args.ear_close_pct
    ear_thresh_semi=EAR_BASE*(args.ear_close_pct+0.10)
    mar_thresh_semi=args.mar_yawn*0.85
    lastEAR=lastMAR=lastYaw=lastPitch=lastRoll=None
    eye_closed_since=None

    # ===============================
    # 6. BUCLE PRINCIPAL
    # ===============================
    import mediapipe as mp
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    prev_time = time.time()
    fps = 0
    frame_times = []
    eye_closed_since = None
    eye_semi_since = None
    mouth_open_since = None

    show_perf = True  # Cambia a False si no quieres ver FPS/latencia

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        estados = []
        color = (255, 255, 255)

        # ===== Tiempo de procesamiento / FPS =====
        now_time = time.time()
        latency_ms = (now_time - prev_time) * 1000
        prev_time = now_time
        frame_times.append(latency_ms)
        if len(frame_times) > 20:
            frame_times.pop(0)
        fps = 1000 / np.mean(frame_times)

        # ===== Cámara tapada o poca luz =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        texture_std = gray.std()
        camera_blocked = texture_std < 8
        low_light = brightness < 40

        if camera_blocked:
            estados.append("Cámara tapada")
        if low_light:
            estados.append("Poca luz")

        # ===== Rostro detectado =====
        lm = detector.detect(frame)
        if lm is not None and not camera_blocked:
            h, w = frame.shape[:2]
            lm_px = lm.copy()
            lm_px[:, 0] *= w
            lm_px[:, 1] *= h

            EAR = compute_EAR(lm_px)
            MAR = compute_MAR(lm_px)
            yaw, pitch, roll = estimate_angles_simple(lm_px)

            lastEAR = ema(lastEAR, EAR)
            lastMAR = ema(lastMAR, MAR)
            lastYaw = ema(lastYaw, yaw)
            lastPitch = ema(lastPitch, pitch)
            lastRoll = ema(lastRoll, roll)

            # ===== OJOS =====
            now = time.time()
            if lastEAR < ear_thresh_closed:
                if eye_closed_since is None:
                    eye_closed_since = now
                if now - eye_closed_since > 1.2:  # cierre largo
                    estados.append("Microsueño / cierre prolongado")
                estados.append("Ojos cerrados")
            elif lastEAR < ear_thresh_semi:
                if eye_semi_since is None:
                    eye_semi_since = now
                if now - eye_semi_since > 2.0:
                    estados.append("Somnolencia leve (ojos semicerrados)")
                estados.append("Ojos medio cerrados")
                eye_closed_since = None
            else:
                eye_closed_since = None
                eye_semi_since = None

            # ===== BOCA =====
            if lastMAR > args.mar_yawn:
                if mouth_open_since is None:
                    mouth_open_since = now
                if now - mouth_open_since > 2.5:
                    estados.append("Bostezo prolongado")
                estados.append("Bostezo")
            elif lastMAR > mar_thresh_semi:
                estados.append("Boca semiabierta")
                mouth_open_since = None
            else:
                mouth_open_since = None

            # ===== CABEZA =====
            pitch_diff = abs(lastPitch - pitch_base)
            yaw_diff = abs(lastYaw - yaw_base)

            if yaw_diff > args.yaw_thr + 5:
                estados.append("Giro lateral")

            # tolerancia de inclinación leve
            if pitch_diff > args.pitch_thr + 10:
                estados.append("Cabeceo / fatiga muscular")
            elif pitch_diff > args.pitch_thr:
                estados.append("Inclinación leve (atención reducida)")


            # ===== Mano tapando boca =====
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = mp_hands.process(rgb)
            hand_on_mouth = False

            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    # Calcular centro y caja de la mano
                    cx = np.mean([pt.x for pt in hand_landmarks.landmark]) * w
                    cy = np.mean([pt.y for pt in hand_landmarks.landmark]) * h
                    minx = min([pt.x for pt in hand_landmarks.landmark]) * w
                    maxx = max([pt.x for pt in hand_landmarks.landmark]) * w
                    miny = min([pt.y for pt in hand_landmarks.landmark]) * h
                    maxy = max([pt.y for pt in hand_landmarks.landmark]) * h

                    # Caja de la boca
                    mouth_top = np.mean(lm_px[[13, 14], 1])
                    mouth_bottom = np.mean(lm_px[[78, 308], 1])
                    mouth_left = lm_px[78, 0]
                    mouth_right = lm_px[308, 0]

                    # Ampliar zona de detección (más tolerante)
                    mouth_top -= 50
                    mouth_bottom += 70
                    mouth_left -= 40
                    mouth_right += 40

                    # Verificar intersección entre caja de mano y boca
                    overlap_x = not (maxx < mouth_left or minx > mouth_right)
                    overlap_y = not (maxy < mouth_top or miny > mouth_bottom)

                    if overlap_x and overlap_y:
                        hand_on_mouth = True
                        break

            if hand_on_mouth:
                estados.append("Mano cubriendo boca")
                if "Bostezo" in estados:
                    estados.append("Bostezo con mano tapando boca")


            # Filtrar puntos válidos dentro del frame
            lm_px = lm_px[(lm_px[:,0] > 0) & (lm_px[:,0] < w) & (lm_px[:,1] > 0) & (lm_px[:,1] < h)]

            print(f"Yaw={lastYaw:.1f}, Pitch={lastPitch:.1f}, Hand={hand_on_mouth}")

            # ===== Visualización =====
            if args.show:
                vis = frame.copy()
                left = lm_px[[33,160,158,133,153,144], :2].astype(int)
                right = lm_px[[362,385,387,263,373,380], :2].astype(int)
                mouth = lm_px[[78,308,14,13,312,82], :2].astype(int)

                for p in left:
                    cv2.circle(vis, tuple(p), 2, (0,255,255), -1)
                for p in right:
                    cv2.circle(vis, tuple(p), 2, (0,255,255), -1)
                for p in mouth:
                    cv2.circle(vis, tuple(p), 2, (255,0,255), -1)

                # Texto principal
                cv2.putText(vis, " | ".join(estados), (18,36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                # Mostrar FPS y latencia opcional
                if show_perf:
                    cv2.putText(vis, f"FPS: {fps:.1f}  Latencia: {latency_ms:.1f} ms",
                                (18,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1)

                cv2.imshow("Antifatiga", cv2.resize(vis, (640,480)))

        else:
            if args.show:
                cv2.imshow("Antifatiga", cv2.resize(frame, (640,480)))

        # ===== JSON OUTPUT =====
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "estados": estados,
            "EAR": round(float(lastEAR or 0),3),
            "MAR": round(float(lastMAR or 0),3),
            "Yaw": round(float(lastYaw or 0),1),
            "Pitch": round(float(lastPitch or 0),1),
            "Roll": round(float(lastRoll or 0),1),
            "fps": round(float(fps),1),
            "latency_ms": round(float(latency_ms),1),
            "camera_blocked": bool(camera_blocked),
            "low_light": bool(low_light),
            "hand_on_mouth": bool(hand_on_mouth)
        }
        publish_event(payload)

        if args.show and (cv2.waitKey(1) & 0xFF == 27):
            break

    mp_hands.close()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
