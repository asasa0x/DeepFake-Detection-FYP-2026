"""
Entrypoint script for Deepfake Detection Backend.
Starts the Flask server for frontend communication and deepfake analysis.
"""

import sys
from pathlib import Path

# Ensure stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from app import app, warmup_detector_async

def start():
    print("=" * 65, flush=True)
    print("[*] Deepfake Detection AI Backend Server", flush=True)
    print(f"    - Server URL        : http://{config.SERVER_HOST}:{config.SERVER_PORT}", flush=True)
    print(f"    - Dataset Directory : {config.DATASET_DIR}", flush=True)
    status = "Custom Trained Weights" if config.MODEL_PATH.exists() else "Pretrained Backbone"
    print(f"    - Model State       : {status}", flush=True)
    print("    - Status            : Server online (background model warmup active)", flush=True)
    print("=" * 65, flush=True)
    
    # Warm up model asynchronously so Flask binds to port 5000 instantly
    warmup_detector_async()
    
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG_MODE)

if __name__ == "__main__":
    start()
