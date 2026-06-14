"""
Flask web app for rural-hospital neonatal jaundice screening.

Features:
- Mobile-first upload/camera capture for skin and sclera photos.
- Baby details capture.
- Skin and sclera region detection previews.
- Preprocessing visualization.
- Dual-input model inference when models/best_model.pth exists.
- Safe heuristic fallback for demos before model training is complete.
- SQLite patient screening records.
- PDF report export for doctors.

This app is for screening workflow development only. Every result page and PDF
includes a reminder to confirm suspected jaundice with a blood test.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import urllib.error
import urllib.request
import uuid
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from torchvision import transforms

import config
from dataset import extract_ycbcr_features
from model import DualInputJaundiceNet
from region_detector import OfflineRegionDetector
from skin_tone_normalizer import SkinToneNormalizer


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "app" / "patients.db"
UPLOAD_DIR = BASE_DIR / "app" / "uploads"
STATIC_GENERATED_DIR = BASE_DIR / "static" / "generated"
MODEL_PATH = config.MODEL_DIR / "best_model.pth"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["SECRET_KEY"] = "replace-with-a-secure-key-before-deployment"

normalizer = SkinToneNormalizer()
region_detector = OfflineRegionDetector()
model_cache: Optional[DualInputJaundiceNet] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_storage() -> None:
    """Create required directories and the SQLite table with migrations."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                baby_name TEXT NOT NULL,
                age_hours REAL NOT NULL,
                birth_weight REAL NOT NULL,
                gestational_age REAL NOT NULL,
                severity TEXT NOT NULL,
                bilirubin REAL NOT NULL,
                recommendation TEXT NOT NULL,
                threshold_status TEXT NOT NULL,
                worsening_alert INTEGER NOT NULL DEFAULT 0,
                skin_path TEXT NOT NULL,
                sclera_path TEXT NOT NULL,
                skin_detection_path TEXT NOT NULL,
                sclera_detection_path TEXT NOT NULL,
                preprocessing_path TEXT NOT NULL,
                gradcam_path TEXT NOT NULL,
                probabilities_json TEXT NOT NULL,
                parent_name TEXT,
                blood_type TEXT,
                notes TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                record_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                synced INTEGER NOT NULL DEFAULT 0,
                synced_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'staff'
            )
            """
        )
        connection.commit()

        # --- MIGRATION CHECK ---
        cursor = connection.execute("PRAGMA table_info(screenings)")
        cols = [c[1] for c in cursor.fetchall()]
        if "parent_name" not in cols:
            connection.execute("ALTER TABLE screenings ADD COLUMN parent_name TEXT")
        if "blood_type" not in cols:
            connection.execute("ALTER TABLE screenings ADD COLUMN blood_type TEXT")
        if "notes" not in cols:
            connection.execute("ALTER TABLE screenings ADD COLUMN notes TEXT")
        connection.commit()


def allowed_file(filename: str) -> bool:
    """Validate image extension."""

    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(image_rgb: np.ndarray, filename: str) -> str:
    """Save an RGB image under static/generated and return its relative URL path."""

    output_path = STATIC_GENERATED_DIR / filename
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)
    return f"generated/{filename}"


def image_to_base64(image_rgb: np.ndarray) -> str:
    """Encode an RGB image as base64 JPEG for REST responses."""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def base64_to_rgb(data: str) -> np.ndarray:
    """Decode a base64 image string into RGB uint8."""

    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Invalid base64 image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detect_skin_region(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect likely skin pixels and draw a region overlay."""

    mask = normalizer.create_skin_mask(image_rgb)
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([46, 204, 113])).astype(np.uint8)
    bordered = draw_largest_contour(overlay, mask, color=(0, 180, 90))
    return mask, bordered


def detect_sclera_region(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect likely sclera pixels from eye photos.

    The mask targets bright, low-saturation white/yellowish areas. It is a
    practical preview aid; production segmentation should use a trained model.
    """

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

    bright_low_saturation = cv2.inRange(
        hsv,
        np.array([0, 0, 80], dtype=np.uint8),
        np.array([60, 95, 255], dtype=np.uint8),
    )
    yellow_white_chroma = cv2.inRange(
        ycrcb,
        np.array([90, 105, 80], dtype=np.uint8),
        np.array([255, 170, 145], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(bright_low_saturation, yellow_white_chroma)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([52, 152, 219])).astype(np.uint8)
    bordered = draw_largest_contour(overlay, mask, color=(20, 110, 220))
    return mask, bordered


def draw_largest_contour(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw the largest detected region on an RGB image."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_rgb.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [largest], -1, color, 3)
    return output


def create_preprocessing_visual(skin_rgb: np.ndarray, sclera_rgb: np.ndarray, token: str) -> str:
    """Create a side-by-side visualization of original and normalized images."""

    skin_norm = normalizer.normalize(skin_rgb)
    sclera_norm = normalizer.normalize(sclera_rgb)
    size = (220, 220)
    panels = [
        cv2.resize(skin_rgb, size),
        cv2.resize(skin_norm, size),
        cv2.resize(sclera_rgb, size),
        cv2.resize(sclera_norm, size),
    ]
    canvas = np.concatenate(panels, axis=1)
    return save_rgb(canvas, f"{token}_preprocessing.jpg")


def get_model() -> Optional[DualInputJaundiceNet]:
    """Load the trained model once, if a checkpoint exists."""

    global model_cache
    if model_cache is not None:
        return model_cache
    if not MODEL_PATH.exists():
        return None

    model = DualInputJaundiceNet(pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model_cache = model
    return model_cache


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB image to normalized EfficientNet input tensor."""

    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )
    pil_image = Image.fromarray(image_rgb)
    return transform(pil_image).unsqueeze(0)


def predict_jaundice(skin_rgb: np.ndarray, sclera_rgb: np.ndarray) -> Dict[str, object]:
    """Run model inference or a deterministic heuristic fallback."""

    model = get_model()
    skin_norm = normalizer.normalize(skin_rgb)
    sclera_norm = normalizer.normalize(sclera_rgb)
    skin_feature_view = cv2.resize(skin_norm, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    sclera_feature_view = cv2.resize(sclera_norm, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    ycbcr_features = extract_ycbcr_features(np.concatenate([skin_feature_view, sclera_feature_view], axis=1))

    if model is None:
        return heuristic_prediction(skin_norm, sclera_norm, ycbcr_features)

    with torch.no_grad():
        skin_tensor = image_to_tensor(skin_norm).to(device)
        sclera_tensor = image_to_tensor(sclera_norm).to(device)
        feature_tensor = torch.tensor(ycbcr_features, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(skin_tensor, sclera_tensor, feature_tensor)
        probabilities = torch.softmax(outputs["logits"], dim=1).cpu().numpy()[0]
        class_index = int(np.argmax(probabilities))
        bilirubin = float(outputs["bilirubin"].cpu().numpy()[0])

    return {
        "severity": config.CLASS_NAMES[class_index],
        "bilirubin": round(bilirubin, 1),
        "probabilities": probabilities.tolist(),
        "source": "model",
    }


def heuristic_prediction(skin_rgb: np.ndarray, sclera_rgb: np.ndarray, features: np.ndarray) -> Dict[str, object]:
    """
    Demo fallback before trained weights exist.

    Uses Cb reduction and yellow excess in skin/sclera. This is intentionally
    labeled as heuristic on the UI and must not be treated as a trained model.
    """

    skin_view = cv2.resize(skin_rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    sclera_view = cv2.resize(sclera_rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    combined = np.concatenate([skin_view, sclera_view], axis=1).astype(np.float32) / 255.0
    red = combined[:, :, 0]
    green = combined[:, :, 1]
    blue = combined[:, :, 2]
    yellow_excess = np.clip(((red + green) / 2.0) - blue, 0, 1)
    cb_mean = float(features[2])
    score = float((yellow_excess.mean() * 18.0) + ((0.55 - cb_mean) * 22.0))
    bilirubin = float(np.clip(4.0 + score, 2.0, 28.0))

    if bilirubin < 5:
        class_index = 0
    elif bilirubin < 12:
        class_index = 1
    elif bilirubin < 20:
        class_index = 2
    else:
        class_index = 3

    probabilities = np.full(config.NUM_CLASSES, 0.08, dtype=np.float32)
    probabilities[class_index] = 0.76

    return {
        "severity": config.CLASS_NAMES[class_index],
        "bilirubin": round(bilirubin, 1),
        "probabilities": probabilities.tolist(),
        "source": "heuristic",
    }


def create_gradcam_like_overlay(image_rgb: np.ndarray, token: str) -> str:
    """
    Create a yellow-region attention overlay.

    When a trained checkpoint is available this can be replaced with true
    GradCAM hooks. For deployment preview, this highlights yellow-dominant
    regions detected from image color channels.
    """

    image = cv2.resize(image_rgb, (420, 420))
    image_float = image.astype(np.float32) / 255.0
    yellow_score = np.clip(((image_float[:, :, 0] + image_float[:, :, 1]) / 2.0) - image_float[:, :, 2], 0, 1)
    yellow_score = cv2.GaussianBlur(yellow_score, (21, 21), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * yellow_score), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.58 * image + 0.42 * heatmap)
    return save_rgb(overlay, f"{token}_gradcam.jpg")


def create_gradcam_like_image(image_rgb: np.ndarray) -> np.ndarray:
    """Return a GradCAM-style yellow-region overlay as an RGB image."""

    image = cv2.resize(image_rgb, (420, 420))
    image_float = image.astype(np.float32) / 255.0
    yellow_score = np.clip(((image_float[:, :, 0] + image_float[:, :, 1]) / 2.0) - image_float[:, :, 2], 0, 1)
    yellow_score = cv2.GaussianBlur(yellow_score, (21, 21), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * yellow_score), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(0.58 * image + 0.42 * heatmap)


def get_age_specific_thresholds(age_hours: float) -> Dict[str, float]:
    """Simplified age-specific bilirubin action thresholds."""

    if age_hours < 24:
        return {"monitor": 8.0, "phototherapy": 10.0, "urgent": 15.0}
    if age_hours < 48:
        return {"monitor": 10.0, "phototherapy": 12.0, "urgent": 18.0}
    if age_hours < 72:
        return {"monitor": 12.0, "phototherapy": 15.0, "urgent": 20.0}
    return {"monitor": 15.0, "phototherapy": 18.0, "urgent": 25.0}


def threshold_status(age_hours: float, bilirubin: float) -> str:
    """Compare bilirubin estimate with age-specific action thresholds."""

    thresholds = get_age_specific_thresholds(age_hours)
    if bilirubin >= thresholds["urgent"]:
        return "Above urgent threshold"
    if bilirubin >= thresholds["phototherapy"]:
        return "Above phototherapy threshold"
    if bilirubin >= thresholds["monitor"]:
        return "Above close-monitoring threshold"
    return "Below age-specific action threshold"


def recommendation_for(severity: str) -> str:
    """Map severity to the requested clinical recommendation."""

    return {
        "Normal": "Continue monitoring",
        "Mild": "Increase feeding frequency",
        "Moderate": "Phototherapy recommended",
        "Severe": "URGENT: Immediate treatment",
    }[severity]


def urgency_for(severity: str) -> str:
    """Map severity to API-friendly urgency."""

    return {
        "Normal": "routine_monitoring",
        "Mild": "low",
        "Moderate": "phototherapy",
        "Severe": "urgent",
    }[severity]


def is_worsening(baby_name: str, bilirubin: float) -> bool:
    """Alert when this baby's latest bilirubin is higher than the previous value."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            """
            SELECT bilirubin FROM screenings
            WHERE lower(baby_name) = lower(?)
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """,
            (baby_name,),
        ).fetchone()
    return bool(row and bilirubin > float(row[0]) + 0.5)


def insert_screening(record: Dict[str, object]) -> int:
    """Insert one screening record and return its database id."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO screenings (
                created_at, baby_name, age_hours, birth_weight, gestational_age,
                severity, bilirubin, recommendation, threshold_status,
                worsening_alert, skin_path, sclera_path, skin_detection_path,
                sclera_detection_path, preprocessing_path, gradcam_path,
                probabilities_json, parent_name, blood_type, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"],
                record["baby_name"],
                record["age_hours"],
                record["birth_weight"],
                record["gestational_age"],
                record["severity"],
                record["bilirubin"],
                record["recommendation"],
                record["threshold_status"],
                int(record["worsening_alert"]),
                record["skin_path"],
                record["sclera_path"],
                record["skin_detection_path"],
                record["sclera_detection_path"],
                record["preprocessing_path"],
                record["gradcam_path"],
                json.dumps(record["probabilities"]),
                record.get("parent_name"),
                record.get("blood_type"),
                record.get("notes"),
            ),
        )
        connection.commit()
        record_id = int(cursor.lastrowid)
    queue_for_sync(record_id, record)
    return record_id


def queue_for_sync(record_id: int, record: Dict[str, object]) -> None:
    """Store a local sync payload for offline-first deployment."""

    payload = {
        "record_id": record_id,
        "created_at": record["created_at"],
        "baby_name": record["baby_name"],
        "age_hours": record["age_hours"],
        "birth_weight": record["birth_weight"],
        "gestational_age": record["gestational_age"],
        "severity": record["severity"],
        "bilirubin": record["bilirubin"],
        "recommendation": record["recommendation"],
        "threshold_status": record["threshold_status"],
        "worsening_alert": bool(record["worsening_alert"]),
        "probabilities": record["probabilities"],
    }
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO sync_queue (created_at, record_id, payload_json, synced)
            VALUES (?, ?, ?, 0)
            """,
            (datetime.datetime.now().isoformat(timespec="seconds"), record_id, json.dumps(payload)),
        )
        connection.commit()


def sync_pending_records(server_url: Optional[str] = None) -> Dict[str, object]:
    """
    Try to sync queued records to a server when Wi-Fi is available.

    Set SYNC_SERVER_URL to the hospital endpoint that accepts POSTed JSON. If no
    URL is configured, the app remains fully local and reports pending records.
    """

    server_url = server_url or os.environ.get("SYNC_SERVER_URL", "").strip()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM sync_queue WHERE synced = 0 ORDER BY id ASC"
        ).fetchall()

    if not server_url:
        return {"configured": False, "pending": len(rows), "synced": 0, "message": "No SYNC_SERVER_URL configured."}

    synced_ids = []
    for row in rows:
        request_data = row["payload_json"].encode("utf-8")
        request_object = urllib.request.Request(
            server_url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request_object, timeout=5) as response:
                if 200 <= response.status < 300:
                    synced_ids.append(int(row["id"]))
        except (urllib.error.URLError, TimeoutError):
            break

    if synced_ids:
        placeholders = ",".join("?" for _ in synced_ids)
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                f"UPDATE sync_queue SET synced = 1, synced_at = ? WHERE id IN ({placeholders})",
                [datetime.now().isoformat(timespec="seconds"), *synced_ids],
            )
            connection.commit()

    return {"configured": True, "pending": len(rows) - len(synced_ids), "synced": len(synced_ids)}


def fetch_record(record_id: int) -> Optional[sqlite3.Row]:
    """Fetch one screening record."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute("SELECT * FROM screenings WHERE id = ?", (record_id,)).fetchone()


def fetch_records() -> list[dict]:
    """Fetch all screening records newest first as dictionaries."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT * FROM screenings ORDER BY datetime(created_at) DESC"
        ).fetchall()]


def trend_for_baby(baby_name: str) -> list[float]:
    """Fetch chronological bilirubin trend for one baby."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT bilirubin FROM screenings
            WHERE lower(baby_name) = lower(?)
            ORDER BY datetime(created_at) ASC
            """,
            (baby_name,),
        ).fetchall()
        return [float(r["bilirubin"]) for r in rows]


@app.route("/", methods=["GET"])
def home():
    """Homepage with upload/camera form."""

    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Simple session-based login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                flash("Logged in successfully!", "success")
                return redirect(url_for("home"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Simple user registration."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        full_name = request.form.get("full_name")
        hashed = generate_password_hash(password)
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                    (username, hashed, full_name),
                )
                conn.commit()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists", "danger")
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.context_processor
def inject_lang():
    return dict(lang=session.get("lang", "en"))


@app.route("/set-lang/<new_lang>")
def set_lang(new_lang):
    if new_lang in ["en", "hi"]:
        session["lang"] = new_lang
    return redirect(request.referrer or url_for("home"))


@app.route("/process", methods=["POST"])
def process_upload():
    """Save uploads, run preprocessing/inference, and show processing page."""

    init_storage()
    skin_file = request.files.get("skin_photo")
    sclera_file = request.files.get("sclera_photo")
    if not skin_file or not sclera_file or not allowed_file(skin_file.filename) or not allowed_file(sclera_file.filename):
        return render_template("index.html", error="Please upload both skin and sclera images."), 400

    token = uuid.uuid4().hex
    skin_path = UPLOAD_DIR / f"{token}_skin{Path(skin_file.filename).suffix.lower()}"
    sclera_path = UPLOAD_DIR / f"{token}_sclera{Path(sclera_file.filename).suffix.lower()}"
    skin_file.save(skin_path)
    sclera_file.save(sclera_path)

    skin_rgb = read_rgb(skin_path)
    sclera_rgb = read_rgb(sclera_path)

    detection = region_detector.analyze(skin_rgb, sclera_rgb)
    skin_detection_path = save_rgb(detection["skin"].overlay_rgb, f"{token}_skin_detection.jpg")
    sclera_detection_path = save_rgb(detection["sclera"].overlay_rgb, f"{token}_sclera_detection.jpg")
    preprocessing_path = create_preprocessing_visual(skin_rgb, sclera_rgb, token)
    gradcam_path = create_gradcam_like_overlay(np.concatenate([cv2.resize(skin_rgb, (420, 420)), cv2.resize(sclera_rgb, (420, 420))], axis=1), token)

    prediction = predict_jaundice(skin_rgb, sclera_rgb)
    baby_name = request.form.get("baby_name", "").strip() or "Unnamed baby"
    age_hours = float(request.form.get("age_hours", 0) or 0)
    birth_weight = float(request.form.get("birth_weight", 0) or 0)
    gestational_age = float(request.form.get("gestational_age", 0) or 0)
    bilirubin = float(prediction["bilirubin"])
    severity = str(prediction["severity"])

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baby_name": baby_name,
        "age_hours": age_hours,
        "birth_weight": birth_weight,
        "gestational_age": gestational_age,
        "severity": severity,
        "bilirubin": bilirubin,
        "recommendation": recommendation_for(severity),
        "threshold_status": threshold_status(age_hours, bilirubin),
        "worsening_alert": is_worsening(baby_name, bilirubin),
        "skin_path": str(skin_path),
        "sclera_path": str(sclera_path),
        "skin_detection_path": skin_detection_path,
        "sclera_detection_path": sclera_detection_path,
        "preprocessing_path": preprocessing_path,
        "gradcam_path": gradcam_path,
        "probabilities": prediction["probabilities"],
        "prediction_source": prediction["source"],
        "parent_name": request.form.get("parent_name", "").strip(),
        "blood_type": request.form.get("blood_type", ""),
        "notes": request.form.get("notes", "").strip(),
    }
    record_id = insert_screening(record)

    return render_template(
        "processing.html",
        record_id=record_id,
        record=record,
        skin_detection_path=skin_detection_path,
        sclera_detection_path=sclera_detection_path,
        preprocessing_path=preprocessing_path,
        detection=detection,
    )


@app.route("/results/<int:record_id>")
def results(record_id: int):
    """Display screening result and recommendation."""

    record = fetch_record(record_id)
    if record is None:
        return redirect(url_for("home"))
    thresholds = get_age_specific_thresholds(float(record["age_hours"]))
    risk_percent = min(max((float(record["bilirubin"]) / thresholds["urgent"]) * 100.0, 0.0), 100.0)
    return render_template(
        "results.html",
        record=record,
        thresholds=thresholds,
        risk_percent=risk_percent,
        probabilities=json.loads(record["probabilities_json"]),
    )


@app.route("/records")
def records():
    """Show all screening history and bilirubin trend data."""

    rows = fetch_records()
    trends = {}
    for row in rows:
        if row["baby_name"] not in trends:
            trends[row["baby_name"]] = trend_for_baby(row["baby_name"])
    return render_template("records.html", records=rows, trends=trends)


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard showing statistics and trends with extra safety."""
    try:
        rows = fetch_records()
        total_screenings = len(rows)
        
        # Stats defaults
        severe_count = 0
        total_bilirubin = 0.0
        patient_count = 0
        daily_counts = [0] * 7
        severity_dist = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
        critical_alerts = []

        if rows:
            # Stats calculation
            severe_count = sum(1 for r in rows if r.get("severity") == "Severe")
            total_bilirubin = sum(float(r.get("bilirubin") or 0) for r in rows)
            patient_count = len(set(str(r.get("baby_name") or "Unknown").lower() for r in rows))

            # Daily counts (last 7 days)
            now = datetime.datetime.now()
            daily_counts = []
            for i in range(6, -1, -1):
                day_str = (now - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                count = sum(1 for r in rows if str(r.get("created_at", "")).startswith(day_str))
                daily_counts.append(count)

            # Severity distribution
            for name in ["Normal", "Mild", "Moderate", "Severe"]:
                count = sum(1 for r in rows if r.get("severity") == name)
                severity_dist[name] = round((count / total_screenings) * 100) if total_screenings > 0 else 0

            # Critical Alerts (latest 5 severe cases)
            critical_alerts = [r for r in rows if r.get("severity") == "Severe"][:5]

        return render_template(
            "dashboard.html",
            total_screenings=total_screenings,
            severe_count=severe_count,
            average_bilirubin=(total_bilirubin / total_screenings) if total_screenings > 0 else 0,
            patient_count=patient_count,
            daily_counts=daily_counts,
            max_count=max(daily_counts) if daily_counts else 1,
            severity_dist=severity_dist,
            critical_alerts=critical_alerts,
        )
    except Exception as e:
        print(f"DASHBOARD ERROR: {e}")
        return render_template(
            "dashboard.html",
            total_screenings=0,
            severe_count=0,
            average_bilirubin=0,
            patient_count=0,
            daily_counts=[0] * 7,
            max_count=1,
            severity_dist={"Normal": 25, "Mild": 25, "Moderate": 25, "Severe": 25},
            critical_alerts=[],
            error_msg=str(e)
        )


@app.route("/about")
def about():
    """About the project and developer Rahul Patel."""
    return render_template("about.html")


@app.route("/how-to-use")
def how_to_use():
    """Step-by-step guide for using the system."""
    return render_template("how_to_use.html")


@app.route("/offline/sync", methods=["GET", "POST"])
def offline_sync():
    """Manual sync endpoint for offline-first deployments."""

    return jsonify(sync_pending_records())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST prediction endpoint.

    Request JSON:
        {
          "skin_image_base64": "...",
          "eye_image_base64": "...",
          "age_hours": 36
        }

    Response JSON:
        severity, bilirubin_estimate, urgency, advice, gradcam_base64
    """

    try:
        payload = request.get_json(force=True)
        skin_rgb = base64_to_rgb(payload["skin_image_base64"])
        eye_rgb = base64_to_rgb(payload["eye_image_base64"])
        age_hours = float(payload.get("age_hours", 0) or 0)
    except Exception as exc:
        return jsonify({"error": f"Invalid request: {exc}"}), 400

    detection = region_detector.analyze(skin_rgb, eye_rgb)
    if not detection["skin"].clear or not detection["sclera"].clear:
        return jsonify(
            {
                "error": "regions_unclear",
                "skin_message": detection["skin"].message,
                "sclera_message": detection["sclera"].message,
                "skin_confidence": detection["skin"].confidence,
                "sclera_confidence": detection["sclera"].confidence,
            }
        ), 422

    prediction = predict_jaundice(skin_rgb, eye_rgb)
    severity = str(prediction["severity"])
    bilirubin = float(prediction["bilirubin"])
    combined = np.concatenate([cv2.resize(skin_rgb, (420, 420)), cv2.resize(eye_rgb, (420, 420))], axis=1)
    gradcam_image = create_gradcam_like_image(combined)

    return jsonify(
        {
            "severity": severity,
            "bilirubin_estimate": round(bilirubin, 1),
            "urgency": urgency_for(severity),
            "advice": recommendation_for(severity),
            "threshold_status": threshold_status(age_hours, bilirubin),
            "probabilities": prediction["probabilities"],
            "prediction_source": prediction["source"],
            "gradcam_base64": image_to_base64(gradcam_image),
            "disclaimer": "For screening only, confirm with blood test",
        }
    )


@app.route("/records/<int:record_id>/pdf")
def export_pdf(record_id: int):
    """Export a doctor-facing PDF report."""

    record = fetch_record(record_id)
    if record is None:
        return redirect(url_for("records"))

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="Neonatal Jaundice Screening Report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Neonatal Jaundice Screening Report", styles["Title"]),
        Spacer(1, 12),
    ]

    details = [
        ["Baby name", record["baby_name"]],
        ["Screened at", record["created_at"]],
        ["Age in hours", f"{record['age_hours']}"],
        ["Birth weight", f"{record['birth_weight']} kg"],
        ["Gestational age", f"{record['gestational_age']} weeks"],
        ["Severity", record["severity"]],
        ["Estimated bilirubin", f"{record['bilirubin']:.1f} mg/dL"],
        ["Threshold comparison", record["threshold_status"]],
        ["Recommendation", record["recommendation"]],
        ["Worsening alert", "Yes" if record["worsening_alert"] else "No"],
    ]
    table = Table(details, colWidths=[150, 330])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccd6dd")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Disclaimer: For screening only, confirm with blood test.",
            styles["Heading3"],
        )
    )
    document.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"jaundice_report_{record_id}.pdf",
        mimetype="application/pdf",
    )


init_storage()


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_ENV", "").lower() == "development"
    app.run(host="0.0.0.0", port=5000, debug=debug_enabled)
