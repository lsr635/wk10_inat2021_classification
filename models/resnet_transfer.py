import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for name, param in self.resnet50.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet50(x)