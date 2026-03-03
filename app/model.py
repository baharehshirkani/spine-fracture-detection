import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SpineModelSingle(nn.Module):
    """
    Spine fracture detection model for single-channel (1 slice) input.
    
    Architecture: EfficientNet-B0 with modified input layer
    Input: (batch, 1, 512, 512)
    Output: (batch, 7) - probabilities for C1-C7 fractures
    """
    
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Modify first conv layer for 1-channel input
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Modify classifier for 7 outputs
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SpineModel3Channel(nn.Module):
    """
    Spine fracture detection model for 3-channel (2.5D) input.
    
    Uses 3 consecutive slices as RGB-like channels.
    Can leverage ImageNet pretrained weights since input is 3-channel.
    
    Architecture: EfficientNet-B0
    Input: (batch, 3, 512, 512)
    Output: (batch, 7) - probabilities for C1-C7 fractures
    """
    
    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Modify classifier for 7 outputs
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(
    model_path: str,
    model_type: str = "3channel",
    device: str = 'cpu'
) -> nn.Module:
    """
    Load a trained spine fracture detection model.
    
    Args:
        model_path: Path to the .pth model file
        model_type: Type of model ('single' or '3channel')
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    if model_type == "single":
        model = SpineModelSingle()
    elif model_type == "3channel":
        model = SpineModel3Channel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def get_model_type_from_path(model_path: str) -> str:
    """
    Infer model type from filename.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Model type string ('single' or '3channel')
    """
    path_lower = model_path.lower()
    if "single" in path_lower:
        return "single"
    elif "3channel" in path_lower or "3_channel" in path_lower or "2.5d" in path_lower:
        return "3channel"
    else:
        # Default to 3channel (better performance)
        return "3channel"
