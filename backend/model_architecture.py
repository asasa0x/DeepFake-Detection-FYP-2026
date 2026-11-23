"""
Deepfake Detection Neural Network Architecture.
Implements an EfficientNet-B0 backbone with a custom forensic classification head.
"""

_DeepfakeClassifierClass = None

def _get_classifier_class():
    global _DeepfakeClassifierClass
    if _DeepfakeClassifierClass is None:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        class _InternalDeepfakeClassifier(nn.Module):
            """
            EfficientNet-B0 based binary classifier for Real vs Fake media detection.
            Classes:
                0: Real
                1: Fake
            """
            def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout_rate: float = 0.4):
                super(_InternalDeepfakeClassifier, self).__init__()
                
                # Load EfficientNet-B0 backbone
                if pretrained:
                    weights = EfficientNet_B0_Weights.DEFAULT
                    self.backbone = efficientnet_b0(weights=weights)
                else:
                    self.backbone = efficientnet_b0(weights=None)
                    
                # Extract features dimension from the original classifier
                in_features = self.backbone.classifier[1].in_features  # 1280 for EfficientNet-B0
                
                # Replace default classifier with custom forensic detection head
                # LayerNorm works flawlessly with any batch size (including batch_size=1)
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(inplace=True),
                    nn.Dropout(p=dropout_rate / 2),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass returning raw logits.
                Output shape: (batch_size, num_classes)
                """
                return self.backbone(x)

            def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass returning softmax probabilities.
                """
                logits = self.forward(x)
                return torch.softmax(logits, dim=1)

        _DeepfakeClassifierClass = _InternalDeepfakeClassifier
    return _DeepfakeClassifierClass


def DeepfakeClassifier(*args, **kwargs):
    cls = _get_classifier_class()
    return cls(*args, **kwargs)


def build_model(pretrained: bool = True, num_classes: int = 2, device = None):
    """
    Instantiate and return the DeepfakeClassifier model moved to the target device.
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = _get_classifier_class()
    model = cls(pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    return model

