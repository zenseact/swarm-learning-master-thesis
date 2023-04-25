import torch.nn as nn
import torchvision.models as models


class MobileNetV2(nn.Module):
    def __init__(self, num_output, **kwargs):
        super(MobileNetV2, self).__init__()
        self.num_output = num_output
        self.model = models.mobilenet_v2(pretrained=True)
        self.is_pretrained = True
        self.model.classifier = nn.Identity()  # remove classification layer
        self.projection_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Dropout(0.2),  # dropout layer
            nn.Flatten(),
            nn.Linear(1280, self.num_output),  # linear projection layer
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.projection_layer(x)
        # reshape to 17 points with 3 coordinates each
        return x.view(-1, 17, 3)

    def model_parameters(self):
        return self.parameters()


# always fill metadata
metadata = dict(
    name="MobileNetV2 with Projection Layer",
    model=MobileNetV2,
)
