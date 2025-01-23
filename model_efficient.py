import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class AgeDetector(nn.Module):
    def __init__(self):
        super(AgeDetector, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
        # Attention layers con dimensioni corrette
        self.attention1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for idx, layer in enumerate(self.efficientnet.features):
            x = layer(x)
            if idx == 0:  # 32 channels
                x = x * self.attention1(x)
            elif idx == 4:  # 80 channels
                x = x * self.attention2(x)
        
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x