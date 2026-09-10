import os
import time
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
CONF_THRESHOLD = 0.25
MIN_DEFECT_SIZE_MM = 5
MAX_DEFECT_SIZE_MM = 500
PORT = 8000

lock = threading.Lock()
latest_frame_jpeg = None
latest_telemetry = {
    "status": "initializing",
    "fps": 0,
    "defects_detected": 0,
    "detections": [],
    "belt_health": 98,
    "has_damage": False,
    "model_loaded": False,
    "conf_threshold": CONF_THRESHOLD,
    "timestamp": time.time()
}

class ConveyorDetector:
    def __init__(self, model_path=MODEL_PATH, conf=CONF_THRESHOLD):
        self.conf = conf
        self.model = None
        self.model_loaded = False
        if HAS_YOLO and os.path.exists(model_path):
            try:
                print(f"[AI Engine] Loading YOLO weights from: {model_path}")
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"[AI Engine] Model loaded successfully! Classes: {self.model.names}")
            except Exception as e:
                print(f"[AI Engine] Warning: Could not load YOLO model: {e}")
        else:
            print(f"[AI Engine] Model file not found at {model_path}. Using fallback detector.")

    def detect_optical_holes_and_objects(self, frame):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        roi = frame[margin_y:h - margin_y, margin_x:w - margin_x]
        if roi.size == 0:
            return [], []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        belt_median = float(np.median(blurred))

        diff = cv2.absdiff(blurred, int(belt_median))
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        dilated = cv2.dilate(cleaned, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_holes = []
        found_objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < (w * h * 0.85):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / bh if bh > 0 else 1.0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                fx1 = bx + margin_x
                fy1 = by + margin_y
                fx2 = fx1 + bw
                fy2 = fy1 + bh

                diameter_est = round((max(bw, bh) / float(w)) * 1200)

                if diameter_est < MIN_DEFECT_SIZE_MM or diameter_est > MAX_DEFECT_SIZE_MM:
                    continue

                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                inner_mean = cv2.mean(blurred, mask=mask)[0]

                is_hole = False
                if abs(inner_mean - belt_median) > 28 and (0.60 < aspect < 1.70 and solidity > 0.65):
                    is_hole = True

                diameter_est = max(5, min(500, diameter_est))

                det_item = {
                    "bbox": [fx1, fy1, fx2, fy2],
                    "norm_bbox": [round(fx1 / w, 3), round(fy1 / h, 3), round(fx2 / w, 3), round(fy2 / h, 3)],
                    "confidence": round(min(0.98, 0.72 + (solidity * 0.26)), 2),
                    "class": "hole" if is_hole else "foreign object",
                    "type": "hole" if is_hole else "foreign_object",
                    "size_mm": diameter_est,
                    "location": f"{'Center' if fx1 > w * 0.35 and fx2 < w * 0.65 else ('Left Edge' if fx1 < w * 0.4 else 'Right Shoulder')}"
                }

                if is_hole:
                    found_holes.append(det_item)
                else:
                    found_objects.append(det_item)

        return found_holes, found_objects

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        detections = []
        has_hole = False
        has_foreign = False
        has_crack = False

        if self.model and self.model_loaded:
            try:
                results = self.model.predict(frame, conf=self.conf, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_name = self.model.names.get(cls_id, "belt damage").lower()

                        bw = b[2] - b[0]
                        bh = b[3] - b[1]

                        if "hole" in raw_name or "puncture" in raw_name:
                            det_type = "hole"
                            cls_display = "HOLE"
                            has_hole = True
                        elif "foreign" in raw_name or "object" in raw_name or "debris" in raw_name:
                            det_type = "foreign_object"
                            cls_display = "FOREIGN OBJECT"
                            has_foreign = True
                        elif "crack" in raw_name or "tear" in raw_name:
                            det_type = "crack"
                            cls_display = "TEAR/CRACK"
                            has_crack = True
                        else:
                            det_type = "foreign_object"
                            cls_display = "SURFACE DAMAGE"
                            has_foreign = True

                        size_est_mm = round((max(bw, bh) / float(w)) * 1200)
                        if size_est_mm < MIN_DEFECT_SIZE_MM:
                            continue

                        detections.append({
                            "bbox": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)],
                            "norm_bbox": [round(b[0] / w, 3), round(b[1] / h, 3), round(b[2] / w, 3), round(b[3] / h, 3)],
                            "confidence": round(conf, 3),
                            "class": cls_display,
                            "type": det_type,
                            "size_mm": size_est_mm,
                            "location": f"{'Center' if b[0] > w * 0.35 and b[2] < w * 0.65 else ('Left Edge' if b[0] < w * 0.4 else 'Right Shoulder')}"
                        })
            except Exception as e:
                print(f"[AI Engine] Predict error: {e}")

        opt_holes, opt_objects = self.detect_optical_holes_and_objects(frame)
        
        for oh in opt_holes:
            overlap = False
            for d in detections:
                if abs(d["bbox"][0] - oh["bbox"][0]) < 50 and abs(d["bbox"][1] - oh["bbox"][1]) < 50:
                    overlap = True
                    break
            if not overlap:
                detections.append(oh)
                has_hole = True

        for oo in opt_objects:
            overlap = False
            for d in detections:
                if abs(d["bbox"][0] - oo["bbox"][0]) < 50 and abs(d["bbox"][1] - oo["bbox"][1]) < 50:
                    overlap = True
                    break
            if not overlap:
                detections.append(oo)
                has_foreign = True

        has_damage = len(detections) > 0

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            dtype = d.get("type", "damage")
            conf = d.get("confidence", 0.9)

            if dtype == "hole":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(12, int(max(x2 - x1, y2 - y1) / 2) + 4)
                cv2.circle(frame, (cx, cy), radius, (0, 0, 245), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 245), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 50, 255), 2)
                label = f"HOLE: {d.get('size_mm', 24)}mm ({conf*100:.0f}%)"
                tag_bg = (0, 0, 220)
            elif dtype == "foreign_object":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                label = f"FOREIGN OBJECT ({conf*100:.0f}%)"
                tag_bg = (0, 140, 230)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 50, 180), 2)
                label = f"TEAR/CRACK ({conf*100:.0f}%)"
                tag_bg = (180, 30, 150)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), tag_bg, -1)
            cv2.putText(frame, label, (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.rectangle(frame, (0, 0), (w, 38), (15, 23, 42), -1)
        if not has_damage:
            status_text = "CONVEYOR SENTINEL LIVE AI  |  BELT NORMAL (NO HOLES / NO FOREIGN OBJECTS)"
            color = (34, 197, 94)
        else:
            alerts = []
            if has_hole:
                alerts.append("HOLE DETECTED")
            if has_foreign:
                alerts.append("FOREIGN OBJECT")
            if has_crack:
                alerts.append("CRACK/TEAR")
            status_text = f"CONVEYOR SENTINEL ALERT: {' + '.join(alerts)}"
            color = (0, 0, 240) if has_hole else (0, 165, 255)

        cv2.putText(frame, status_text, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, detections, has_hole, has_foreign, has_crack

def camera_loop():
    global latest_frame_jpeg, latest_telemetry
    detector = ConveyorDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Camera] Warning: Could not open webcam 0. Creating synthetic conveyor feed...")
        use_synthetic = True
    else:
        use_synthetic = False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[Camera] Webcam initialized successfully!")

    fps_counter = 0
    fps_timer = time.time()
    current_fps = 30

    synthetic_offset = 0

    while True:
        t_start = time.time()
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
        else:
            frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
            cv2.rectangle(frame, (100, 120), (1180, 600), (55, 60, 68), -1)
            synthetic_offset = (synthetic_offset + 8) % 80
            for gx in range(120 - synthetic_offset, 1180, 80):
                cv2.line(frame, (gx, 120), (gx, 600), (40, 45, 52), 2)
            cv2.putText(frame, "SYNTHETIC BELT BENCHMARK (WEBCAM BUSY OR STANDBY)", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            time.sleep(0.03)

        processed_frame, detections, has_hole, has_foreign, has_crack = detector.process_frame(frame)
        has_damage = len(detections) > 0

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        _, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg_bytes = jpeg.tobytes()

        health = 98 - (len(detections) * 35) if has_damage else 98
        health = max(15, min(100, health))

        with lock:
            latest_frame_jpeg = jpeg_bytes
            latest_telemetry = {
                "status": "running",
                "fps": current_fps,
                "defects_detected": len(detections),
                "detections": detections,
                "belt_health": health,
                "has_damage": has_damage,
                "has_hole": has_hole,
                "has_foreign_object": has_foreign,
                "has_crack": has_crack,
                "model_loaded": detector.model_loaded,
                "conf_threshold": detector.conf,
                "timestamp": time.time()
            }

        elapsed = time.time() - t_start
        if elapsed < 0.033:
            time.sleep(0.033 - elapsed)

class SentinelHTTPHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            dash_path = os.path.join(os.path.dirname(__file__), 'conveyor-sentinel-dashboard.html')
            if os.path.exists(dash_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(dash_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Conveyor Sentinel AI Server Running.")
                return

        elif self.path == '/telemetry':
            with lock:
                payload = json.dumps(latest_telemetry).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(payload)
            return

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.end_headers()

            while True:
                with lock:
                    frame = latest_frame_jpeg
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.04)
            return

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server():
    server = ThreadingHTTPServer(('0.0.0.0', PORT), SentinelHTTPHandler)
    print(f"[*] Server running on: http://localhost:{PORT}")
    print(f"[*] Telemetry API:    http://localhost:{PORT}/telemetry")
    print(f"[*] Live Video Stream: http://localhost:{PORT}/stream")
    print(f"[*] Dashboard UI:      http://localhost:{PORT}/")
    server.serve_forever()

if __name__ == '__main__':
    cam_t = threading.Thread(target=camera_loop, daemon=True)
    cam_t.start()
    run_server()
