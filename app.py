"""
app.py — Flask web application for Hard Hat Detection
Run: python app.py
Requirements: pip install -r requirements.txt
"""

import os
import uuid
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from predict_utils import load_model, predict_image

# ── App Configuration ────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

MODEL_PATH = "ppe_best.pt"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Load Model ───────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
if model is None:
    print("[FATAL] Could not load model. Exiting.")
    exit(1)
print("Model ready.\n")


# ── Helpers ──────────────────────────────────────────────────────────────

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Validate file presence
    if "image" not in request.files:
        return jsonify(success=False, error="No image file provided"), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(success=False, error="No file selected"), 400

    if not allowed_file(file.filename):
        return jsonify(success=False, error="Invalid file type. Use JPG, PNG, BMP, or WebP."), 400

    # Save uploaded file with unique name
    ext = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = UPLOAD_DIR / unique_name

    try:
        file.save(str(upload_path))
    except Exception as e:
        return jsonify(success=False, error=f"Failed to save file: {str(e)}"), 500

    # Run prediction
    result_filename = f"result_{unique_name}"
    result_path = RESULTS_DIR / result_filename

    result = predict_image(
        img_path=str(upload_path),
        model=model,
        output_path=str(result_path),
    )

    # Clean up uploaded file
    try:
        upload_path.unlink(missing_ok=True)
    except Exception:
        pass

    if not result["success"]:
        return jsonify(success=False, error=result.get("error", "Prediction failed")), 500

    # Build response matching frontend API contract
    # Capitalize class names for consistent display (model uses lowercase)
    detections = []
    for det in result["detections"]:
        detections.append({
            "class_name": det["class"].capitalize(),
            "confidence": round(det["confidence"], 4),
            "bbox": [round(v, 1) for v in det["bbox"]],
        })

    counts = {k.capitalize(): v for k, v in result["counts"].items()}

    return jsonify(
        success=True,
        detections=detections,
        counts=counts,
        violation=result["violation"],
        image_url=f"/results/{result_filename}",
    )


@app.route("/results/<path:filename>")
def get_result(filename):
    return send_from_directory("results", filename)


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
    )


# ── Cleanup old results periodically ────────────────────────────────────

@app.route("/cleanup", methods=["POST"])
def cleanup_old_files():
    """Remove result files older than 1 hour."""
    removed = 0
    cutoff = time.time() - 3600
    for d in [UPLOAD_DIR, RESULTS_DIR]:
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass
    return jsonify(removed=removed)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Hard Hat Detector web app...")
    print("Open http://localhost:5001 in your browser.\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
