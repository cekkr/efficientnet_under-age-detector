from torch import nn
from torchvision.models import mobilenet_v2

class AgeDetector(nn.Module):
    def __init__(self):
        super(AgeDetector, self).__init__()

        # Carica il modello base
        self.backbone = mobilenet_v2(pretrained=True)

        # Modifica il primo layer convoluzionale per accettare immagini più grandi
        # Questo manterrà i pesi pre-addestrati ma adatterà il modello a input più grandi
        original_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            original_layer.in_channels,
            original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=False
        )

        # Copia i pesi dal layer originale
        self.backbone.features[0][0].weight.data = original_layer.weight.data

        # Modifica il classificatore per l'output binario
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.backbone.last_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)