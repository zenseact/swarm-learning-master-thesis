from static_params import *

class PT_Model(pl.LightningModule):
    def __init__(self, cid=0) -> None:
        super(PT_Model, self).__init__()
        
        self.model = None
        if(c('model') == 'resnet18'):
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif(c('model') == 'mobile_net'):
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        self.change_head_net()
        self.useEma = c('use_ema')
        self.ema = None

        if(self.useEma):
            self.ema = EMA(
                self.model,
                beta=0.995,
                update_after_step=100,
                power=3/4,
                inv_gamma=1.0
            )

        self.is_pretrained = True
        self.loss_fn = torch.nn.L1Loss()
        self.cid = cid
        
        # pytorch imagenet calculated mean/std
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
        self.epoch_counter = 1
        self.tb_log = {}
        self.create_intermediate_steps()
        

    def forward(self, image):
        label = self.model(image)

        if(self.useEma):
            ema_label = self.ema(image)
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
            
    def shared_step(self, batch, batch_idx, stage, inter_outputs, ema_inter_outputs):
        image = batch["image"]
        label = batch["label"]

        logits_label, ema_logits_label = self.forward(image)
        logits_label = logits_label.unsqueeze(dim=1)
        loss = self.loss_fn(logits_label, label)

        if(self.useEma):
            ema_logits_label = ema_logits_label.unsqueeze(dim=1)
            ema_loss = self.loss_fn(ema_logits_label, label)

        ema_loss = ema_loss if(self.useEma) else None
        
        inter_outputs.append(loss.item())
        if(self.useEma):
            ema_inter_outputs.append(ema_loss.item())
            if(stage == 'train'):
                self.ema.update()

        return loss

    def shared_epoch_end(self, inter_outputs, ema_inter_outputs, stage):
        self.tb_log[f'{stage}_loss'] = np.mean(inter_outputs)
        
        if(self.useEma):
            self.tb_log[f'{stage}_ema_loss'] = np.mean(ema_inter_outputs) 

        self.log_dict(self.tb_log, prog_bar=True)
        if(stage == 'train'):
            self.updateTB(self.tb_log, stage)
            self.epoch_counter +=1

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train', self.inter_train_outputs, self.inter_train_ema_outputs)
        return loss

    def on_train_epoch_end(self):
        return self.shared_epoch_end(self.inter_train_outputs, self.inter_train_ema_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'valid', self.inter_val_outputs, self.inter_val_ema_outputs)
        return loss

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.inter_val_outputs, self.inter_val_ema_outputs, 'valid')

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'test', self.inter_test_outputs, self.inter_test_ema_outputs)
        return loss

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.inter_test_outputs, self.inter_test_ema_outputs, 'test')

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

    def updateTB(self, tb_log, stage):
        writer.add_scalars('Loss', tb_log, global_step=self.epoch_counter)
        self.create_intermediate_steps()

    def create_intermediate_steps(self):
        self.inter_train_outputs = []
        self.inter_train_ema_outputs = []

        self.inter_val_outputs = []
        self.inter_val_ema_outputs = []

        self.inter_test_outputs = []
        self.inter_test_ema_outputs = []
        self.tb_log = {}

    def set_TB_loggers(self, logger):
        self.epoch_logger = logger