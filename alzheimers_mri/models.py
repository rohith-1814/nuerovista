from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .constants import CLASS_NAMES, MODEL_NAMES


def build_model(name: str, num_classes: int = len(CLASS_NAMES), pretrained: bool = False):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}")
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_checkpoint(checkpoint, model_name: str, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Checkpoint must contain a 'state_dict' from train.py.")
    if payload.get("model_name") != model_name:
        raise ValueError("Selected model does not match the checkpoint architecture.")
    class_to_idx = payload.get("class_to_idx")
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected:
        raise ValueError(
            "Checkpoint classes do not match the required labels and ordering: "
            + ", ".join(CLASS_NAMES)
        )
    model = build_model(model_name)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def checkpoint_path(model_name: str) -> Path:
    return Path("models") / f"{model_name}.pth"
