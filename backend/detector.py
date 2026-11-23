import io
import time
import threading
from pathlib import Path
from typing import Union, Dict, Any
from PIL import Image

import config
from model_architecture import build_model


class DeepfakeDetector:
    """
    Production Deepfake Detection pipeline powered by EfficientNet-B0 forensics.
    Uses trained neural network weights for direct inference with temperature calibration.
    """
    def __init__(self, model_path: Union[str, Path] = config.MODEL_PATH, device = None):
        self.device = device if device is not None else config.get_device()
        self.model_path = Path(model_path)
        self.model = None
        self.is_custom_trained = False
        self.transform = None

        self._load_model()

    def _load_model(self):
        """
        Loads trained model weights if present, or initializes pretrained backbone.
        """
        import torch
        from torchvision import transforms

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ])

        has_custom_weights = self.model_path.exists()
        self.model = build_model(pretrained=not has_custom_weights, num_classes=2, device=self.device)
        
        if has_custom_weights:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.is_custom_trained = True
                print(f"[Detector] Loaded custom trained weights from {self.model_path}", flush=True)
            except Exception as e:
                print(f"[Detector] Warning: Could not load custom weights ({e}). Using initialized backbone.", flush=True)
        else:
            print(f"[Detector] No custom weights at {self.model_path}. Initialized EfficientNet-B0 backbone.", flush=True)

        self.model.eval()

    def validate_image(self, file_source: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Image.Image:
        """
        Validates the input image format, structure, and integrity.
        Returns a valid PIL Image in RGB format.
        """
        try:
            if isinstance(file_source, (str, Path)):
                img = Image.open(file_source)
            elif isinstance(file_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(file_source))
            elif isinstance(file_source, io.BytesIO):
                file_source.seek(0)
                img = Image.open(file_source)
            elif isinstance(file_source, Image.Image):
                img = file_source
            else:
                raise ValueError("Unsupported image input type.")

            img.verify()  # Verify integrity
            
            # Reopen after verify (verify closes stream/data)
            if isinstance(file_source, (str, Path)):
                img = Image.open(file_source).convert("RGB")
            elif isinstance(file_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(file_source)).convert("RGB")
            elif isinstance(file_source, io.BytesIO):
                file_source.seek(0)
                img = Image.open(file_source).convert("RGB")
            else:
                img = img.convert("RGB")

            # Check dimensions
            w, h = img.size
            if w < 10 or h < 10:
                raise ValueError(f"Image is too small ({w}x{h}). Minimum required is 10x10.")

            return img
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")

    def predict(self, file_source: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Dict[str, Any]:
        """
        Performs deepfake detection on the uploaded image.
        Returns a dictionary with result ('REAL' or 'FAKE') and confidence percentage.
        Class 0 = Real, Class 1 = Fake
        """
        import torch
        import numpy as np

        start_time = time.time()
        
        # 1. Validate & convert image
        img_pil = self.validate_image(file_source)

        # 2. Apply transformations
        tensor_img = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 3. Model Inference directly from the trained neural network
        with torch.no_grad():
            logits = self.model(tensor_img).cpu().numpy()[0]
            
            # Temperature calibration (T=0.50) expands dynamic range while preserving model decisions
            temp = 0.50
            scaled_logits = logits / temp
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            real_prob = float(probs[0])
            fake_prob = float(probs[1])

        # 4. Determine result and confidence percentage
        real_prob_pct = round(real_prob * 100.0, 2)
        fake_prob_pct = round(fake_prob * 100.0, 2)

        if fake_prob >= real_prob:
            result = "FAKE"
            confidence = round(fake_prob * 100.0, 1)
        else:
            result = "REAL"
            confidence = round(real_prob * 100.0, 1)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "result": result,
            "confidence": confidence,
            "real_probability": real_prob_pct,
            "fake_probability": fake_prob_pct,
            "latency_ms": latency_ms,
            "model": "EfficientNet-B0-Forensics",
            "is_custom_trained": self.is_custom_trained
        }


# Thread-safe singleton instance for fast API requests
_detector_instance = None
_detector_lock = threading.Lock()

def is_detector_loaded() -> bool:
    """
    Returns True if the singleton detector instance has completed initialization.
    """
    return _detector_instance is not None and _detector_instance.model is not None

def get_detector() -> DeepfakeDetector:
    """
    Returns the singleton DeepfakeDetector instance using thread-safe initialization.
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = DeepfakeDetector()
    return _detector_instance
