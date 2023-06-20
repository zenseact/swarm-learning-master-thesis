import torch
import torch.nn as nn
import numpy as np


class HPBaseline(nn.Module):
    def __init__(self, **kwargs):
        super(HPBaseline, self).__init__()
        self.target_distances = kwargs.get(
            "target_distances",
            [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165],
        )
        arr = np.zeros((17, 3))
        arr[:, 0] = self.target_distances
        self.static_prediction = nn.Parameter(torch.tensor(arr, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.size(0)
        static_prediction = self.static_prediction.expand(batch_size, -1, -1)
        return static_prediction.view(batch_size, -1)

    def model_parameters(self):
        return self.parameters()


# always fill metadata
metadata = dict(
    name="MobileNetV2 with Projection Layer",
    model=HPBaseline,
)
