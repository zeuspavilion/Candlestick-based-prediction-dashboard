import torch
import torch.nn as nn
import torchvision.models as models

class CandlestickCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_vit_model(num_classes: int = 3, freeze_pretrained_backbone: bool = True):
    weights_enum = getattr(models, "ViT_B_16_Weights", None)
    weights = getattr(weights_enum, "DEFAULT", None) if weights_enum is not None else None
    
    try:
        model = models.vit_b_16(weights=weights)
        pretrained = weights is not None
    except Exception:
        model = models.vit_b_16(weights=None)
        pretrained = False

    if pretrained and freeze_pretrained_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(in_features, num_classes),
    )
    return model, pretrained

def build_resnet_model(num_classes: int = 3, freeze_pretrained_backbone: bool = False):
    weights_enum = getattr(models, "ResNet18_Weights", None)
    weights = getattr(weights_enum, "DEFAULT", None) if weights_enum is not None else None
    
    try:
        model = models.resnet18(weights=weights)
        pretrained = weights is not None
    except Exception:
        model = models.resnet18(weights=None)
        pretrained = False

    if pretrained and freeze_pretrained_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model, pretrained

def build_cnn_model(num_classes: int = 3, freeze_pretrained_backbone: bool = False):
    return CandlestickCNN(num_classes=num_classes), False

MODEL_BUILDERS = {
    "vit_b_16": build_vit_model,
    "resnet18": build_resnet_model,
    "custom_cnn": build_cnn_model,
}
