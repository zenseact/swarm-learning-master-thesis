import logging
import segmentation_models_pytorch as smp
import torch

logger = logging.getLogger(__name__)


def build_unet():
    logger.info(f"Model: UNet with EfficientNet-B4 encoder")
    model = FixMatchSeg()
    return model


# Use variable img_size for auto import from data config


class FixMatchSeg(torch.nn.Module):
    def __init__(
        self,
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        img_size=None,
        **kwargs,
    ):
        logger.info(
            f"Creating FixMatchSeg model with {encoder_name} encoder, {encoder_weights} weights, {in_channels} input channels, {classes} classes"
        )
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

    def forward(self, x):
        logits_mask = self.unet(x)
        return logits_mask.sigmoid()

    def model_parameters(self):
        return self.parameters()


# always fill metadata
metadata = dict(
    name="UNet with EfficientNet-B4 encoder",
    model=FixMatchSeg,
)
