"""
Configuration module for Deepfake Detection Backend.
Provides path resolution, hardware acceleration selection,
and server settings for the deepfake detection service.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
DATASET_DIR = Path(os.getenv("DF_DATASET_DIR", BASE_DIR / "DF-DATASET" / "Cleaned"))

def get_dataset_subfolder(base_path: Path, target_names: list[str]) -> Path:
    if not base_path.exists():
        return base_path / target_names[0]
    for child in base_path.iterdir():
        if child.is_dir() and child.name.lower() in [n.lower() for n in target_names]:
            return child
    return base_path / target_names[0]

REAL_DIR = get_dataset_subfolder(DATASET_DIR, ["Real", "real", "REAL"])
FAKE_DIR = get_dataset_subfolder(DATASET_DIR, ["Fake", "fake", "FAKE"])

# Model paths
MODEL_DIR = BASE_DIR / "model"
PRIMARY_MODEL_PATH = MODEL_DIR / "experiments" / "efficientnet_b0_new_best.pth"
INFERENCE_MODEL_PATH = MODEL_DIR / "experiments" / "efficientnet_b0_new_best_inference.pth"
MODEL_PATH = Path(
    os.getenv(
        "DF_MODEL_PATH",
        PRIMARY_MODEL_PATH if PRIMARY_MODEL_PATH.exists() else INFERENCE_MODEL_PATH,
    )
)

# Server configuration (0.0.0.0 + PORT for Render compatibility)
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", 5000))
DEBUG_MODE = os.getenv("FLASK_DEBUG", "0") == "1"

# Upload and security parameters
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload size

# Hardware Device Selection (Lazy-loaded to keep startup instantaneous)
_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

def __getattr__(name: str):
    if name == "DEVICE":
        return get_device()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Image preprocessing standards
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)
