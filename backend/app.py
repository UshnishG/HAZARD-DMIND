import os
import cv2
import base64
import tempfile
import sqlite3
import threading
import json
import uuid
import numpy as np
from datetime import datetime
import os
import cv2
import base64
import tempfile
import sqlite3
import threading
import json
import uuid
import numpy as np
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import supervision as sv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

def secure_filename(filename: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

API_KEY             = os.environ.get("ROBOFLOW_API_KEY")
LANDSLIDE_PROJECT   = "segformer-landslide-detection"
LANDSLIDE_VERSION   = 2
LANDMINE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
FIRE_MODEL_PATH     = os.path.join(BASE_DIR, "fire.pt")
HUMAN_MODEL_PATH    = os.path.join(BASE_DIR, "human.pt")

VIDEO_FRAME_SKIP = 3

FIRE_COLORS = {
    "LOW":      (0,  210, 255),
    "MODERATE": (0,  120, 255),
    "HIGH":     (0,   40, 255),
}
LANDSLIDE_COLORS = {
    "LOW":      (0,  230, 230),
    "MODERATE": (0,  140, 220),
    "HIGH":     (0,   80, 200),
}
LANDMINE_COLOR = (50, 205, 50)
HUMAN_COLOR    = (0, 255, 0)
SEV_RANK       = {"LOW": 0, "MODERATE": 1, "HIGH": 2}

# ==============================
# DATABASE
# ==============================

DB_PATH = os.path.join(BASE_DIR, "mission.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS missions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   TEXT    NOT NULL,
            ended_at     TEXT,
            source_file  TEXT,
            total_frames INTEGER DEFAULT 0,
            threat_count INTEGER DEFAULT 0,
            status       TEXT    DEFAULT 'active'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id         INTEGER REFERENCES missions(id),
            detected_at        TEXT    NOT NULL,
            det_type           TEXT    NOT NULL,
            severity           TEXT,
            frame_index        INTEGER,
            timestamp_in_video TEXT,
            source_file        TEXT,
            confidence         REAL
        )
    """)
    con.commit()
    con.close()

init_db()

def db_start_mission(source_file):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO missions (started_at, source_file, status) VALUES (?,?,?)",
                (datetime.utcnow().isoformat(), source_file, "active"))
    mid = cur.lastrowid
    con.commit(); con.close()
    return mid

def db_end_mission(mission_id, total_frames, threat_count):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE missions SET ended_at=?, total_frames=?, threat_count=?, status=? WHERE id=?",
                (datetime.utcnow().isoformat(), total_frames, threat_count, "complete", mission_id))
    con.commit(); con.close()

def db_log_event(mission_id, det_type, severity, frame_index=None, ts_video=None, source_file="", confidence=None):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO events (mission_id, detected_at, det_type, severity, frame_index,
                   timestamp_in_video, source_file, confidence) VALUES (?,?,?,?,?,?,?,?)""",
                (mission_id, datetime.utcnow().isoformat(), det_type, severity,
                 frame_index, ts_video, source_file, confidence))
    con.commit(); con.close()

def db_get_recent_events(limit=100):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def db_get_missions(limit=20):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM missions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def db_stats():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    total_missions = cur.execute("SELECT COUNT(*) FROM missions").fetchone()[0]
    total_frames   = cur.execute("SELECT SUM(total_frames) FROM missions").fetchone()[0] or 0
    threat_count   = cur.execute("SELECT COUNT(*) FROM events WHERE det_type != 'SAFE'").fetchone()[0]
    
    fire_count      = cur.execute("SELECT COUNT(*) FROM events WHERE det_type = 'FIRE'").fetchone()[0]
    landslide_count = cur.execute("SELECT COUNT(*) FROM events WHERE det_type = 'LANDSLIDE'").fetchone()[0]
    landmine_count  = cur.execute("SELECT COUNT(*) FROM events WHERE det_type = 'LANDMINE'").fetchone()[0]
    human_count     = cur.execute("SELECT COUNT(*) FROM events WHERE det_type = 'HUMAN'").fetchone()[0]
    con.close()
    
    return {
        "total_missions": total_missions,
        "total_frames": total_frames,
        "threat_count": threat_count,
        "fire_count": fire_count,
        "landslide_count": landslide_count,
        "landmine_count": landmine_count,
        "human_count": human_count,
        "last_updated": datetime.utcnow().isoformat()
    }

# ==============================
# LOAD MODELS
# ==============================
from huggingface_hub import hf_hub_download

from inference_sdk import InferenceHTTPClient

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "LFQ3tQ1G6KvnwISSoIFW")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "ieees-workspace-lez6m")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "landslide-segmentation-api-1783455950308")

print("Loading Landslide Model (Roboflow Workflow)...")
try:
    rf_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    print("Landslide workflow client loaded")
except Exception as e:
    rf_client = None
    print("Roboflow client load error:", e)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "Ushnish2004/LandmineModel")
if not HF_REPO_ID:
    print("WARNING: HF_REPO_ID environment variable is not set. Models will not be downloaded from Hugging Face.")
    
def get_model_path(local_path, filename):
    if HF_REPO_ID:
        try:
            print(f"Fetching {filename} from Hugging Face ({HF_REPO_ID})...")
            return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        except Exception as e:
            print(f"Error downloading {filename} from HF: {e}")
    return local_path

print("Loading Landmine YOLO model...")
landmine_path = get_model_path(LANDMINE_MODEL_PATH, "best.pt")
if os.path.exists(landmine_path):
    landmine_model = YOLO(landmine_path)
    print("Landmine model loaded")
else:
    landmine_model = None
    print("Landmine model file missing:", landmine_path)

print("Loading Fire Segmentation model...")
fire_path = get_model_path(FIRE_MODEL_PATH, "fire.pt")
if os.path.exists(fire_path):
    fire_model = YOLO(fire_path)
    fire_model.model.fuse = lambda verbose=True: fire_model.model
    print("Fire model loaded")
else:
    fire_model = None
    print("Fire model file missing:", fire_path)

print("Loading Human Detection model...")
human_path = get_model_path(HUMAN_MODEL_PATH, "human.pt")
if os.path.exists(human_path):
    human_model = YOLO(human_path)
    print("Human model loaded from", human_path)
else:
    print(f"Human model not found at {human_path} - falling back to yolov8l.pt")
    human_model = YOLO("yolov8l.pt")

box_annotator = sv.BoxAnnotator(thickness=3)

# ==============================
# HELPERS
# ==============================

def unique_filename(filename):
    stem, ext = os.path.splitext(secure_filename(filename))
    return f"{stem}_{uuid.uuid4().hex[:8]}{ext}"

def to_bgr(frame):
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()

def is_colour_image(frame):
    b, g, r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    return not (
        np.mean(np.abs(b.astype(int) - g.astype(int))) < 3 and
        np.mean(np.abs(g.astype(int) - r.astype(int))) < 3
    )

def severity_from_area(perc):
    if perc >= 20: return "HIGH"
    if perc >= 5:  return "MODERATE"
    return "LOW"

def alpha_from_severity(sev):
    return {"HIGH": 0.65, "MODERATE": 0.50, "LOW": 0.30}[sev]

def draw_filled_poly(canvas, poly, color, alpha):
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly], color)
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

def clean_severity(sev):
    return sev if sev in SEV_RANK else "MODERATE"

# ==============================
# DETECTION FUNCTIONS
# ==============================

def detect_humans(frame, annotated):
    if human_model is None:
        return annotated, False, 0
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = human_model(rgb, conf=0.40, verbose=False)[0]
    count   = 0
    for box in results.boxes:
        if int(box.cls[0]) != 0: continue
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), HUMAN_COLOR, 2)
        label = f"Human  {conf:.0%}"
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale, thick = 0.60, 1
        (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
        lx = max(x1, 2)
        ly = max(y1 - th - 10, th + 4)
        # Remove the solid black label box for a cleaner overlay; keep text readable with an outline.
        #cv2.putText(annotated, label, (lx, ly), font, scale, (0, 0, 0), thick+1, cv2.LINE_AA)
        #cv2.putText(annotated, label, (lx, ly), font, scale, HUMAN_COLOR, thick, cv2.LINE_AA)
    if count > 0:
        badge = f"Humans: {count}"
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale, thick = 0.70, 2
        (tw, th), _ = cv2.getTextSize(badge, font, scale, thick)
        pad = 8
        h, w = annotated.shape[:2]
        bx, by = w - tw - pad*2 - 4, 4
        cv2.rectangle(annotated, (bx, by), (bx+tw+pad*2, by+th+pad*2), (15,15,15), -1)
        cv2.putText(annotated, badge, (bx+pad, by+th+pad-2), font, scale, HUMAN_COLOR, thick, cv2.LINE_AA)
    return annotated, count > 0, count


def detect_fire(frame, annotated):
    if fire_model is None:
        return annotated, False, ""
    fire_found = False
    max_sev    = "LOW"
    h, w       = frame.shape[:2]
    results    = fire_model(frame, conf=0.25, verbose=False)
    for result in results:
        if result.masks is not None:
            for poly in result.masks.xy:
                poly = np.array(poly, dtype=np.int32)
                if len(poly) < 3: continue
                fire_found = True
                perc  = (cv2.contourArea(poly) / (w * h)) * 100
                sev   = severity_from_area(perc)
                if SEV_RANK[sev] > SEV_RANK.get(max_sev, -1):
                    max_sev = sev
                color = FIRE_COLORS[sev]
                alpha = alpha_from_severity(sev)
                annotated = draw_filled_poly(annotated, poly, color, alpha)
                cv2.polylines(annotated, [poly], True, color, 3)
                cv2.polylines(annotated, [poly], True, (255,255,255), 1)
        elif result.boxes is not None:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                fire_found = True
                area = (x2 - x1) * (y2 - y1)
                perc = (area / (w * h)) * 100
                sev = severity_from_area(perc)
                if SEV_RANK[sev] > SEV_RANK.get(max_sev, -1):
                    max_sev = sev
                color = FIRE_COLORS[sev]
                alpha = alpha_from_severity(sev)
                # Draw filled rectangle for blending
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return annotated, fire_found, max_sev if fire_found else ""


def detect_landslide(frame, annotated, image_path):
    if rf_client is None:
        return annotated, False, "", False, ""
    h, w            = frame.shape[:2]
    landslide_found = False
    severity_str    = ""
    rf_fire_found   = False
    rf_fire_sev     = ""
    
    try:
        # Run workflow via inference_sdk
        result = rf_client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={
                "image": image_path
            },
            use_cache=True
        )
        
        # 1. Parse predictions to determine severity
        predictions = []
        output_dict = {}
        if isinstance(result, list) and len(result) > 0:
            output_dict = result[0]
        elif isinstance(result, dict):
            output_dict = result

        # Check fire-proofing output
        if output_dict.get("fire_detected") is True:
            rf_fire_found = True
            fire_preds = output_dict.get("fire_predictions", [])
            total_fire_area = 0
            if isinstance(fire_preds, list):
                for fp in fire_preds:
                    total_fire_area += fp.get("width", 0) * fp.get("height", 0)
            if total_fire_area > 0:
                perc = (total_fire_area / (w * h)) * 100
                rf_fire_sev = severity_from_area(perc)
            else:
                rf_fire_sev = "LOW"

        # Extract predictions list
        if "predictions" in output_dict:
            inner = output_dict["predictions"]
            if isinstance(inner, dict) and "predictions" in inner:
                predictions = inner["predictions"]
            elif isinstance(inner, list):
                predictions = inner
        
        # Calculate severity based on bounding box areas if predictions exist
        if predictions:
            total_area = 0
            for pred in predictions:
                # Ensure we only count predictions actually labeled as landslide
                pred_class = str(pred.get("class", "")).lower()
                if "fire" in pred_class:
                    continue  # Skip false positives or fire detections
                
                pred_w = pred.get("width", 0)
                pred_h = pred.get("height", 0)
                total_area += (pred_w * pred_h)
            
            if total_area > 0 and not rf_fire_found:
                landslide_found = True
                perc = (total_area / (w * h)) * 100
                sev = severity_from_area(perc)
                severity_str = f"LANDSLIDE  {sev}"
            
        # 2. Extract and overlay the output_image (mask)
        overlay_data = output_dict.get("output_image")
        if overlay_data:
            if isinstance(overlay_data, dict):
                overlay_data = overlay_data.get("value") or overlay_data.get("base64") or overlay_data.get("data")
            
            if isinstance(overlay_data, str):
                if overlay_data.startswith("data:image"):
                    overlay_data = overlay_data.split(",", 1)[1]
                elif overlay_data.startswith("http://") or overlay_data.startswith("https://"):
                    import requests
                    resp = requests.get(overlay_data, timeout=10)
                    if resp.status_code == 200:
                        overlay_data = base64.b64encode(resp.content).decode("utf-8")
                    else:
                        overlay_data = None
                
                if overlay_data:
                    # Decode base64 to numpy array
                    image_bytes = base64.b64decode(overlay_data)
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    mask_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    
                    if mask_img is not None:
                        mask_img = cv2.resize(mask_img, (w, h))
                        # If it's a 4-channel image (RGBA/BGRA), overlay using alpha channel
                        if mask_img.shape[-1] == 4:
                            alpha = mask_img[:, :, 3] / 255.0
                            for c in range(3):
                                annotated[:, :, c] = (alpha * mask_img[:, :, c] + (1 - alpha) * annotated[:, :, c])
                        else:
                            # If 3-channel, extract mask pixels using absdiff against original frame
                            diff = cv2.absdiff(frame, mask_img[:, :, :3])
                            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                            _, thresh = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_BINARY)
                            # Copy only the mask overlay pixels at 100% opacity, preserving other annotations
                            annotated[thresh > 0] = mask_img[:, :, :3][thresh > 0]

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Landslide SDK workflow error:", e)
    return annotated, landslide_found, severity_str, rf_fire_found, rf_fire_sev


def detect_landmine(frame, annotated):
    if landmine_model is None:
        return annotated, False, 0
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results    = landmine_model(rgb_frame, conf=0.25, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections) == 0:
        return annotated, False, 0
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    return annotated, True, len(detections)

# ==============================
# SHARED FRAME ANNOTATION
# ==============================

def annotate_frame(frame, image_path):
    frame     = to_bgr(frame)
    annotated = frame.copy()
    parts     = []
    annotated, human_found, human_count = detect_humans(frame, annotated)
    if human_found:
        parts.append(("HUMAN", f"x{human_count}"))
    if is_colour_image(frame):
        annotated, fire_found, fire_sev = detect_fire(frame, annotated)
        if fire_found:
            parts.append(("FIRE", fire_sev))
        else:
            annotated, ls_found, ls_label, rf_fire, rf_fire_sev = detect_landslide(frame, annotated, image_path)
            if ls_found:
                sev = ls_label.split()[-1] if ls_label else "LOW"
                parts.append(("LANDSLIDE", sev))
            if rf_fire:
                parts.append(("FIRE", rf_fire_sev))
    else:
        annotated, lm_found, lm_count = detect_landmine(frame, annotated)
        if lm_found:
            parts.append(("LANDMINE", f"x{lm_count}"))
    return annotated, parts

# ==============================
# IMAGE PIPELINE  (FIXED)
# ==============================

def process_image(image_path, filename):
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, "Failed to load image â€” unsupported format or corrupt file.", []

    mission_id       = db_start_mission(filename)
    annotated, parts = annotate_frame(raw, image_path)

    # â”€â”€ FIX 1: always write as .jpg for universal browser support â”€â”€
    stem            = os.path.splitext(filename)[0]
    result_filename = f"processed_{stem}.jpg"
    result_path     = os.path.join(RESULT_FOLDER, result_filename)

    # â”€â”€ FIX 2: ensure annotated is proper 3-channel BGR before saving â”€â”€
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    elif annotated.shape[2] == 4:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)

    success = cv2.imwrite(result_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return None, "Failed to save processed image.", []

    label_map = {
        "HUMAN":     lambda s: f"HUMAN  {s}",
        "FIRE":      lambda s: f"FIRE  {s}",
        "LANDSLIDE": lambda s: f"LANDSLIDE  {s}",
        "LANDMINE":  lambda s: f"LANDMINE  {s}",
    }
    status_strs  = [label_map.get(t, lambda s: f"{t} {s}")(s) for t, s in parts]
    status_text  = ("  |  ".join(status_strs) + "  DETECTED") if status_strs else "SAFE â€” NO THREATS DETECTED"
    threat_count = 1 if parts else 0

    for (dtype, sev) in parts:
        cs = clean_severity(sev)
        db_log_event(mission_id, dtype, cs, frame_index=0, source_file=filename)

    db_end_mission(mission_id, 1, threat_count)
    return result_filename, status_text, parts

# ==============================
# VIDEO PIPELINE  (FIXED)
# ==============================

def _frame_to_timestamp(idx, fps):
    total = int(idx / max(fps, 1))
    return f"{total//3600:02d}:{(total%3600)//60:02d}:{total%60:02d}"

def _reencode_h264(src):
    """Re-encode to browser-compatible H.264 with pix_fmt yuv420p."""
    dst = src.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{src}" -vcodec libx264 -crf 23 -preset fast '
        f'-movflags +faststart -pix_fmt yuv420p -an "{dst}" -loglevel error'
    )
    if ret == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1024:
        os.remove(src)
        os.rename(dst, src)
    else:
        if os.path.exists(dst):
            os.remove(dst)
    return src

def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], 0, 0

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # â”€â”€ FIX 3: H.264 requires even dimensions â”€â”€
    width  = width  if width  % 2 == 0 else width  - 1
    height = height if height % 2 == 0 else height - 1

    stem            = os.path.splitext(filename)[0]
    result_filename = f"processed_{stem}.mp4"
    result_path     = os.path.join(RESULT_FOLDER, result_filename)

    out = cv2.VideoWriter(
        result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )
    if not out.isOpened():
        cap.release()
        return None, [], 0, 0

    mission_id     = db_start_mission(filename)
    detections_log = []
    threat_count   = 0
    frame_idx      = 0
    last_annotated = None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            # â”€â”€ FIX 4: resize to even dimensions so VideoWriter doesn't corrupt frames â”€â”€
            frame = cv2.resize(frame, (width, height))

            if frame_idx % VIDEO_FRAME_SKIP == 0 or frame_idx == 1:
                cv2.imwrite(tmp_path, frame)
                annotated, parts = annotate_frame(frame, tmp_path)
                last_annotated   = annotated
                if parts:
                    threat_count += 1
                    ts = _frame_to_timestamp(frame_idx, fps)
                    for (dtype, sev) in parts:
                        cs = clean_severity(sev)
                        detections_log.append({"type": dtype, "severity": cs,
                                               "filename": f"frame_{frame_idx:05d}", "time": ts})
                        db_log_event(mission_id, dtype, cs, frame_index=frame_idx,
                                     ts_video=ts, source_file=filename)
            else:
                annotated = last_annotated if last_annotated is not None else frame

            out.write(annotated)
    finally:
        cap.release()
        out.release()
        try: os.remove(tmp_path)
        except OSError: pass

    # â”€â”€ FIX 5: re-encode for browser playback â”€â”€
    _reencode_h264(result_path)
    db_end_mission(mission_id, frame_idx, threat_count)
    return result_filename, detections_log, frame_idx, threat_count

# ==============================
# ALLOWED EXTENSIONS
# ==============================

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in IMAGE_EXTENSIONS

def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS

# ==============================
# API ROUTES
# ==============================

@app.get("/api/events")
def api_events(limit: int = 100):
    return db_get_recent_events(limit)

@app.get("/api/missions")
def api_missions(limit: int = 20):
    return db_get_missions(limit)

@app.get("/api/stats")
def api_stats():
    return db_stats()

@app.get("/api/queue")
def api_queue():
    return {"queue_depth": 0, "worker_alive": True}

@app.post("/api/detection/image")
async def detection_image(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"error": "No file received."}, status_code=400)
    if not allowed_image(file.filename):
        return JSONResponse({"error": "Unsupported image type."}, status_code=400)

    filename = unique_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_filename, status_text, parts = process_image(filepath, filename)
    if not result_filename:
        return JSONResponse({"error": status_text}, status_code=500)

    return {
        "result_url": f"/static/results/{result_filename}",
        "original_url": f"/static/uploads/{filename}",
        "status_text": status_text,
        "detections": [{"type": t, "severity": s} for t, s in parts]
    }

@app.post("/api/detection/video")
async def detection_video(video: UploadFile = File(...)):
    if not video.filename:
        return JSONResponse({"error": "No file received."}, status_code=400)
    if not allowed_video(video.filename):
        return JSONResponse({"error": "Unsupported video type."}, status_code=400)

    filename = unique_filename(video.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    result_filename, detections_log, total_frames, threat_count = process_video(filepath, filename)
    if not result_filename:
        return JSONResponse({"error": "Video processing failed."}, status_code=500)

    return {
        "video_result_url": f"/static/results/{result_filename}",
        "video_original_url": f"/static/uploads/{filename}",
        "detections": detections_log,
        "frame_count": total_frames,
        "threat_count": threat_count
    }

# Mount frontend for local development/testing convenience
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
