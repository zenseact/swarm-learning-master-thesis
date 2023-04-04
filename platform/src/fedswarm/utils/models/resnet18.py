import numpy as np

from torch import Tensor, nn
from torchvision import models

# Use variable img_size for auto import from data config

class Resnet18(nn.Module):
    def __init__(self, /, num_output, **kwargs) -> None:
        super(Resnet18, self).__init__()
        self.num_output = num_output
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.is_pretrained = True

        """freeze parameters and replace head"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.change_head_net()

    def forward(self, x: Tensor) -> Tensor:
        return np.squeeze(self.model(x))

    def model_parameters(self):
        return self.get_head().parameters()

    def change_head_fc(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_output)

    def change_head_net(self):
        num_ftrs = self.model.fc.in_features
        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_output, bias=True),
        )
        self.model.fc = head_net

    def get_head(self):
        return self.model.fc


# always fill metadata
metadata = dict(
    name="Resnet18 Large Head",
    model=Resnet18,
)
