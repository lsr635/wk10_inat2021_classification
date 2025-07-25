import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from .attention import ChannelAttention

class ResNet50Transfer(nn.Module):
    
    def __init__(self, num_classes: int = 150, dropout_rate: float = 0.6, attention_ratio: int = 16):
        super(ResNet50Transfer, self).__init__()
        
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self._freeze_backbone()
        
        self.attention = ChannelAttention(2048, ratio=attention_ratio)
        
        num_features = self.resnet50.fc.in_features
        self.classifier = self._build_classifier(num_features, num_classes, dropout_rate)
        self.resnet50.fc = nn.Identity()
    
    def _freeze_backbone(self) -> None:
        for param in self.resnet50.parameters():
            param.requires_grad = False
    
    def _build_classifier(self, num_features: int, num_classes: int, dropout_rate: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.8),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        att = self.attention(x)
        x = x * att
        
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        
        return self.classifier(x)
    
    def unfreeze_layer(self, layer_name: str) -> None:
        for name, param in self.resnet50.named_parameters():
            if name.startswith(layer_name):
                param.requires_grad = True