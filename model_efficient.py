import torch.nn as nn
from torchvision.models import efficientnet_b0

class AgeDetector(nn.Module):
    def __init__(self):
        super(AgeDetector, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        
        # Modifica l'ultimo layer per output singolo (probabilità minorenne)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
        # Aggiungi attention layer dopo alcune feature maps
        self.attention1 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get intermediate features
        features = []
        for idx, layer in enumerate(self.efficientnet.features):
            x = layer(x)
            if idx in [4, 7]:  # Dopo i primi blocchi MBConv
                attention = self.attention1(x) if idx == 4 else self.attention2(x)
                x = x * attention
                features.append(x)
        
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        
        return x