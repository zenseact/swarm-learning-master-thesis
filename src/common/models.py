from common.static_params import global_configs

# from common.static_params import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# import lightning.pytorch as
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.change_head_net()

        self.loss_fn = torch.nn.L1Loss()

        self.dists = torch.Tensor(global_configs.TARGET_DISTANCES).to(
            global_configs.DEVICE
        )

    def forward(self, image):
        r = self.model(image).reshape((-1, 17, 3))
        r[:] *= self.dists[:, None]
        r = r.reshape((-1, 3 * 17))
        return r

    def model_parameters(self):
        return self.model.parameters()

    def change_head_net(self):
        num_ftrs = self.model.classifier[-1].in_features

        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 51, bias=True),
        )

        self.model.classifier[-1] = head_net

    def compute_metrics(pred_trajectory, target_trajectory):
        # L1 and L2 distance: matrix of size BSx40x3
        L1_loss = torch.abs(pred_trajectory - target_trajectory)
        L2_loss = torch.pow(pred_trajectory - target_trajectory, 2)

        # BSx40x3 -> BSx3 average over the predicted points
        L1_loss = L1_loss.mean(axis=1)
        L2_loss = L2_loss.mean(axis=1)

        # split into losses for each axis and an avg loss across 3 axes
        # All returned tensors have shape (BS)
        return {
            "L1_loss": L1_loss.mean(axis=1),
            "L1_loss_x": L1_loss[:, 0],
            "L1_loss_y": L1_loss[:, 1],
            "L1_loss_z": L1_loss[:, 2],
            "L2_loss": L2_loss.mean(axis=1),
            "L2_loss_x": L2_loss[:, 0],
            "L2_loss_y": L2_loss[:, 1],
            "L2_loss_z": L2_loss[:, 2],
        }


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.is_pretrained = False
        stride = 1
        nr_cv = 2
        # get the dimentions correct
        size_before_fc = global_configs.IMG_SIZE
        for i in range(0, nr_cv):
            size_before_fc = (size_before_fc - 2 * stride) // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(size_before_fc * size_before_fc * 64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, global_configs.NUM_OUTPUT),
        )

    def forward(self, x):
        return np.squeeze(self.conv(x))

    def model_parameters(self):
        return self.parameters()


class PTNet(nn.Module):
    def __init__(self) -> None:
        super(PTNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.is_pretrained = True

        """freeze parameters and replace head"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.change_head_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return np.squeeze(self.model(x))

    def model_parameters(self):
        return self.get_head().parameters()

    def change_head_fc(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, global_configs.NUM_OUTPUT)

    def change_head_net(self):
        num_ftrs = self.model.fc.in_features
        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, global_configs.NUM_OUTPUT, bias=True),
        )
        self.model.fc = head_net

    def get_head(self):
        return self.model.fc
