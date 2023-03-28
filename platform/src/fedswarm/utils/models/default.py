import numpy as np

from torch import nn

# Use variable img_size for auto import from data config



class Net(nn.Module):
    def __init__(self, num_output, **kwargs):
        super(Net, self).__init__()
        stride = 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=stride, padding=0
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
            
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=stride, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            
            nn.Flatten(),
            nn.Linear(10240, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_output),
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
