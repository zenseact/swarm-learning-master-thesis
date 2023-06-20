"""
This file contains custom PyTorch data transforms for image datasets.

Custom transforms can be defined by creating a new class that inherits from `torchvision.transforms.Transform` and implementing
the `__call__` method. The `__call__` method should apply the desired transformation to the input image and return the transformed
image.

To use a custom transform, create a `transforms.Compose` object that includes the custom transform along with any other desired
transforms, and pass this object to the `transform` argument when creating a PyTorch dataset.

For information about how to work with the platform datasets and custom transforms. See the example notebook
example-notebooks/CustomTransforms.ipynb.

For more information on creating custom transforms, see the PyTorch documentation:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

from torchvision.transforms.functional import crop


class CustomCrop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)
