import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class AgeDetector(nn.Module):
    def __init__(self):
        super(AgeDetector, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x