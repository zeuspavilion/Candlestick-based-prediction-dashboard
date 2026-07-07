import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from MarketPulse.config import (
    IMAGE_SIZE,
    PROJECT_ROOT,
    DEVICE,
    LABELS,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
    MODEL_DIR,
)
from MarketPulse.ml.models import MODEL_BUILDERS

logger = logging.getLogger("marketpulse.ml.inference")

imagenet_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        # Normalization
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits.detach()

def get_gradcam_target_layer(model_name: str, model: nn.Module):
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "custom_cnn":
        return model.features[-4]
    return None

def gradient_saliency(model: nn.Module, input_tensor: torch.Tensor, class_idx: int = None):
    model.zero_grad(set_to_none=True)
    saliency_input = input_tensor.clone().detach().requires_grad_(True)
    logits = model(saliency_input)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())
    logits[:, class_idx].sum().backward()
    saliency = saliency_input.grad.detach().abs().max(dim=1)[0].squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency, logits.detach()

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored_heatmap, 0, 1)

class CandlestickPredictor:
    def __init__(self, checkpoint_path: str = None):
        self.model_name = None
        self.model = None
        self.checkpoint = None
        
        # Resolve best checkpoint dynamically if not provided
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
            
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            logger.warning("No model checkpoint found. Prediction features will run in mock mode.")

    def _find_best_checkpoint(self) -> Path:
        pths = list(Path(MODEL_DIR).glob("*.pth"))
        if pths:
            # Pick first available or return None
            return pths[0]
        return None

    def _load_checkpoint(self, path: str):
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
            
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model_name = checkpoint["model_name"]
        config = checkpoint["config"]
        
        builder = MODEL_BUILDERS.get(self.model_name)
        model, _ = builder(num_classes=3, freeze_pretrained_backbone=False)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(DEVICE)
        model.eval()
        
        self.model = model
        self.checkpoint = checkpoint

    def predict(self, image_path: Path) -> dict:
        """Runs model inference on raw image file."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load and transform image
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = imagenet_transform(img_pil).to(DEVICE)
        
        if self.model is None:
            # Mock mode: return random predictions with deterministic seed
            h = hash(image_path.name) % 3
            probs = [0.1, 0.1, 0.1]
            probs[h] = 0.8
            label = LABELS[h]
            return {
                "label": label,
                "confidence": 0.8,
                "probabilities": probs,
                "mocked": True,
            }

        # Run inference
        input_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx = probs.argmax()
            label = IDX_TO_CLASS[pred_idx]
            
        return {
            "label": label,
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist(),
            "mocked": False,
        }

    def explain(self, image_path: Path) -> dict:
        """Generates prediction and explainability heatmap overlay."""
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = imagenet_transform(img_pil)
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        if self.model is None:
            # Return empty or dummy explainability
            image = np.array(img_pil.resize(IMAGE_SIZE)) / 255.0
            return {
                "image": image,
                "heatmap": np.zeros(IMAGE_SIZE),
                "overlay": image,
                "method": "Mock",
                "label": "neutral",
                "confidence": 0.5,
            }

        target_layer = get_gradcam_target_layer(self.model_name, self.model)
        if target_layer is not None:
            cam = GradCAM(self.model, target_layer)
            heatmap, logits = cam(input_tensor)
            cam.remove_hooks()
            method = "Grad-CAM"
        else:
            heatmap, logits = gradient_saliency(self.model, input_tensor)
            method = "Gradient Saliency"

        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = probabilities.argmax()
        label = IDX_TO_CLASS[pred_idx]
        
        image = denormalize_image(img_tensor)
        overlay = overlay_heatmap(image, heatmap)

        return {
            "image": image,
            "heatmap": heatmap,
            "overlay": overlay,
            "method": method,
            "label": label,
            "confidence": float(probabilities[pred_idx]),
        }
