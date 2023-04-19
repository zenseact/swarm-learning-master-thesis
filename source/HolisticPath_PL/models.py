from static_params import *

class PT_Model(pl.LightningModule):
    def __init__(self) -> None:
        super(PT_Model, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.is_pretrained = True

        # freeze parameters and replace head
        for param in self.model.parameters():
            param.requires_grad = False
        self.change_head_net()
        self.loss_fn = torch.nn.L1Loss()
        
        # pytorch imagenet calculated mean/std
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
        self.inter_train_outputs = []
        self.inter_val_outputs = []
        self.inter_test_outputs = []

    def forward(self, image):
        # normalize image here
        mean = torch.tensor(self.mean).view(3, 1, 1).to(DEVICE)
        std = torch.tensor(self.std).view(3, 1, 1).to(DEVICE)
        
        image = (image - mean) / std
        label = self.model(image)
        return label

    def model_parameters(self):
        return self.model.fc.parameters()

    def change_head_net(self):
        num_ftrs = self.model.fc.in_features
        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, NUM_OUTPUT, bias=True),
        )
        self.model.fc = head_net

    
    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        label = batch["label"]

        logits_label = self.forward(image)
        logits_label = logits_label.unsqueeze(dim=1)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_label, label)

        return {
            "loss": loss,
        }

    def shared_epoch_end(self, outputs, stage):
        metrics = {
            f"{stage}_loss": outputs[-1]['loss'],
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        self.inter_train_outputs.append(output)
        return output

    def on_training_epoch_end(self):
        return self.shared_epoch_end(self.inter_train_outputs, "train")

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")
        self.inter_val_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.inter_val_outputs, "valid")

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch, "test")
        self.inter_test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.inter_test_outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)