from static_params import *

class PT_Model(pl.LightningModule):
    def __init__(self) -> None:
        super(PT_Model, self).__init__()

        self.model = None
        if(c('model') == 'resnet18'):
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif(c('model') == 'mobile_net'):
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        self.change_head_net()
        self.ema = None

        if(c('use_ema')):
            self.ema = EMA(
                self.model,
                beta=0.995,
                update_after_step=100,
                power=3/4,
                inv_gamma=1.0
            )

        self.is_pretrained = True
        self.loss_fn = torch.nn.L1Loss()
        
        # pytorch imagenet calculated mean/std
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
        self.epoch_counter = 1

        self.train_epoch_start_batch_idx = 0
        self.val_epoch_start_batch_idx = 0

        self.inter_train_outputs = []
        self.inter_train_ema_outputs = []

        self.inter_val_outputs = []
        self.inter_val_ema_outputs = []

        self.inter_test_outputs = []
        self.inter_test_ema_outputs = []

    def forward(self, image):
        label = self.model(image)

        if(c('use_ema')):
            ema_label = self.ema(image)
            print(c('use_ema'))
            return label, ema_label

        return label, None

    def model_parameters(self):
        return self.model.fc.parameters()

    def change_head_net(self):
        num_ftrs = 0

        if(c('model') == 'resnet18'):
            num_ftrs = self.model.fc.in_features
        elif(c('model') == 'mobile_net'):
            num_ftrs = self.model.classifier[-1].in_features

        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, c('output_size'), bias=True),
        )

        if(c('model') == 'resnet18'):
            self.model.fc = head_net
        elif(c('model') == 'mobile_net'):
            self.model.classifier[-1] = head_net
            
    def shared_step(self, batch, stage):
        image = batch["image"]
        
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        label = batch["label"]

        logits_label, ema_logits_label = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        logits_label = logits_label.unsqueeze(dim=1)
        loss = self.loss_fn(logits_label, label)

        if(c('use_ema')):
            ema_logits_label = ema_logits_label.unsqueeze(dim=1)
            ema_loss = self.loss_fn(ema_logits_label, label)

        return {
            "loss": loss,
            "ema_loss": ema_loss if(c('use_ema')) else None
        }

    def shared_epoch_end(self, outputs, ema_outputs, stage):
        metrics = {
            f"{stage}_loss": outputs[-1],
            f"{stage}_ema_loss": ema_outputs[-1] if(c('use_ema')) else None,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        
        if(batch_idx == 1 and len(self.inter_train_outputs) > 2):
            epoch_loss = np.mean(self.inter_train_outputs[self.train_epoch_start_batch_idx:]) 
            print(f'\nstarted new train epoch. Last epoch batch indexes: {self.train_epoch_start_batch_idx}-{len(self.inter_train_outputs)}. Train loss: {epoch_loss}')
            writer.add_scalars(TB_CENTRALIZED_SUB_PATH + "epoch", {"train": epoch_loss},self.epoch_counter)
            self.train_epoch_start_batch_idx = len(self.inter_train_outputs)
            
            if(c('use_ema')):
                ema_epoch_loss = np.mean(self.inter_train_ema_outputs[self.train_epoch_start_batch_idx:]) 
                writer.add_scalars(TB_CENTRALIZED_SUB_PATH + "epoch", {"train_ema": ema_epoch_loss},self.epoch_counter)

            
        self.inter_train_outputs.append(output['loss'].item())
        if(c('use_ema')):
            self.inter_train_ema_outputs.append(output['ema_loss'].item())
            # Update the EMA model after each training step
            self.ema.update()

        return output

    def on_training_epoch_end(self):
        self.shared_epoch_end(self.inter_train_outputs, self.inter_train_ema_outputs, "train")

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")

        if(batch_idx == 1 and len(self.inter_val_outputs) > 2):
            epoch_loss = np.mean(self.inter_val_outputs[self.val_epoch_start_batch_idx:]) 
            print(f'\nstarted new val epoch. Last epoch batch indexes: {self.val_epoch_start_batch_idx}-{len(self.inter_val_outputs)}. Val loss: {epoch_loss}')
            writer.add_scalars(TB_CENTRALIZED_SUB_PATH + "epoch", {"val": epoch_loss}, self.epoch_counter)
            self.val_epoch_start_batch_idx = len(self.inter_val_outputs)

            if(c('use_ema')):
                ema_epoch_loss = np.mean(self.inter_val_ema_outputs[self.val_epoch_start_batch_idx:]) 
                writer.add_scalars(TB_CENTRALIZED_SUB_PATH + "epoch", {"ema_val": ema_epoch_loss}, self.epoch_counter)
            
            # update epoch counter only after validation step
            self.epoch_counter +=1

        self.inter_val_outputs.append(output['loss'].item())
        if(c('use_ema')):
            self.inter_val_ema_outputs.append(output['ema_loss'].item())

        return output

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.inter_val_outputs, self.inter_val_ema_outputs, "valid")

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch, "test")

        self.inter_test_outputs.append(output['loss'].item())
        if(c('use_ema')):
            self.inter_test_ema_outputs.append(output['ema_loss'].item())
        return output

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.inter_test_outputs, self.inter_test_ema_outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=c('learning_rate'))
    
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
                'L1_loss':   L1_loss.mean(axis=1),
                'L1_loss_x': L1_loss[:, 0],
                'L1_loss_y': L1_loss[:, 1],
                'L1_loss_z': L1_loss[:, 2],
                'L2_loss':   L2_loss.mean(axis=1),
                'L2_loss_x': L2_loss[:, 0],
                'L2_loss_y': L2_loss[:, 1],
                'L2_loss_z': L2_loss[:, 2]}