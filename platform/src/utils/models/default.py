import numpy as np

from torch import nn

# Use variable img_size for auto import from data config


class Net(nn.Module):
    def __init__(self, /, img_size, num_output, **kwargs):
        super(Net, self).__init__()
        self.is_pretrained = False
        stride = 1
        nr_cv = 2
        # get the dimentions correct
        if type(img_size) is int:
            size_before_fc = [img_size, img_size]
        else:
            size_before_fc = [img_size[0], img_size[1]]

        for _ in range(0, nr_cv):
            size_before_fc[0] = (size_before_fc[0] - 2 * stride) // 2
            size_before_fc[1] = (size_before_fc[1] - 2 * stride) // 2

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
            nn.Linear(size_before_fc[0] * size_before_fc[1] * 64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_output),
        )

    def forward(self, x):
        return np.squeeze(self.conv(x))

    def model_parameters(self):
        return self.parameters()


# always fill metadata
metadata = dict(
    name="DefaultNet",
    model=Net,
)
