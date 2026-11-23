"""
Flask API Application for Deepfake Detection System.
Provides REST API endpoints for image deepfake analysis and health status.
Fully compatible with the Vanilla JavaScript Frontend.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import threading
import config
from detector import get_detector, is_detector_loaded

FRONTEND_DIR = config.BASE_DIR.parent / "Frontend"

# Initialize Flask application (serves Frontend static files directly on :5000 as well)
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Enable CORS for all routes (allows Frontend requests from any port / file://)
CORS(app, resources={r"/*": {"origins": "*"}})

def warmup_detector_async():
    """
    Spawns a background daemon thread to load PyTorch & model weights
    concurrently without blocking the Flask server startup.
    """
    thread = threading.Thread(target=get_detector, name="ModelWarmupDaemon", daemon=True)
    thread.start()
    return thread


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint serving the Frontend UI (index.html).
    """
    if (FRONTEND_DIR / "index.html").is_file():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({
        "status": "online",
        "service": "Deepfake Detection AI Engine",
        "backbone": "EfficientNet-B0 Forensics",
        "device": str(config.DEVICE),
        "model_loaded": is_detector_loaded(),
        "endpoint": "POST /detect"
    }), 200


@app.route("/api/status", methods=["GET"])
def api_status():
    """
    JSON service status endpoint.
    """
    return jsonify({
        "status": "online",
        "service": "Deepfake Detection AI Engine",
        "backbone": "EfficientNet-B0 Forensics",
        "device": str(config.DEVICE),
        "model_loaded": is_detector_loaded(),
        "endpoint": "POST /detect"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint returning system status without blocking.
    """
    loaded = is_detector_loaded()
    is_custom = False
    if loaded:
        d = get_detector()
        is_custom = d.is_custom_trained
    elif config.MODEL_PATH.exists():
        is_custom = True

    return jsonify({
        "status": "healthy",
        "device": str(config.DEVICE),
        "model_loaded": loaded,
        "is_custom_trained": is_custom,
        "dataset_exists": config.DATASET_DIR.exists()
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    """
    Main detection endpoint consumed by Frontend (script.js).
    Accepts multipart/form-data with 'image' field.
    Returns:
        {
            "result": "REAL" | "FAKE",
            "confidence": float (50.0 - 99.9),
            "real_probability": float,
            "fake_probability": float,
            "latency_ms": float,
            "model": "EfficientNet-B0-Forensics"
        }
    """
    try:
        if "image" not in request.files and "file" not in request.files:
            return jsonify({"error": "No image file provided. Please upload an image using the 'image' field."}), 400

        file = request.files.get("image") or request.files.get("file")

        if not file or file.filename == "":
            return jsonify({"error": "No file selected. Please select a valid image."}), 400

        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type '{ext}'. Allowed formats: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
            }), 400

        # Read image content
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Perform Deepfake Detection
        detector = get_detector()
        prediction = detector.predict(image_bytes)

        # Return exact JSON structure expected by the Frontend
        return jsonify({
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "real_probability": prediction["real_probability"],
            "fake_probability": prediction["fake_probability"],
            "latency_ms": prediction["latency_ms"],
            "model": prediction["model"]
        }), 200

    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as err:
        app.logger.error(f"Detection error: {err}", exc_info=True)
        return jsonify({"error": f"Detection failed: {str(err)}"}), 500


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File size exceeds maximum permitted limit (32MB)"}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error occurred"}), 500


if __name__ == "__main__":
    print(f"[*] Deepfake Detection Backend Server running on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG_MODE)
