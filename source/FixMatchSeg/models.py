from static_params import *
from fixmatch_utils import *

class PTModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.ema = EMA(
            self.model,
            beta=0.995,
            update_after_step=100,
            power=3/4,
            inv_gamma=1.0
        )
        
        # for image segmentation dice loss could be the best first choice
        self.DiceLoss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.epoch_counter = 1
        self.tb_log = {}
        self.create_intermediate_steps()

    def forward(self, image):
        # normalize image here
        #image = (image - self.mean) / self.std
        mask = self.model(image)
        ema_mask = self.ema(image)
    
        return mask.sigmoid(), ema_mask.sigmoid()

    def shared_step_fixmatch(self, batch, stage, outputs):
        image = batch["image"]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        isLabeled = batch['isLabeled']
        labeled_idx = torch.where(isLabeled == True)[0].tolist()
        unlabeled_idx = [i for i in range(len(isLabeled)) if i not in labeled_idx]
        
        is_all_labeled = len(labeled_idx) == len(isLabeled)
        is_all_unlabeled = len(unlabeled_idx) == len(isLabeled)

        combinedLoss = None
        combinedEmaLoss = None

        if(stage != 'train'):
            logits_mask, ema_mask = self.forward(image)
            combinedLoss = self.DiceLoss(logits_mask, mask)
            combinedEmaLoss = self.DiceLoss(ema_mask, mask)
            
            self.log_dict({f"{stage} supervised loss": combinedLoss.item()}, prog_bar=True)

        else:
            loss_u = 0; loss = 0; ema_loss = 0

            image_u_w = batch['image_u_w']
            image_u_s = batch['image_u_s']

            image = image[labeled_idx]
            mask = mask[labeled_idx]

            image_u_w = image_u_w[unlabeled_idx]
            image_u_s = image_u_s[unlabeled_idx]

            logits_mask, ema_mask = self.forward(image)

            #logits_mask = binary_mask(logits_mask)
            #ema_mask = binary_mask(ema_mask)

            if(not is_all_labeled):
                logits_mask_u_w, _ = self.forward(image_u_w)
                logits_mask_u_s, _ = self.forward(image_u_s)

                loss_u = compute_unsupervised_loss(logits_mask_u_w, logits_mask_u_s, THRESHOLD)
                self.log_dict({f"{stage} unsupervised loss": loss_u.item()}, prog_bar=True)
            
            if(not is_all_unlabeled):
                #loss = compute_supervised_loss(mask, logits_mask)
                #ema_loss = compute_supervised_loss(mask, ema_mask)
                
                loss = self.DiceLoss(mask, logits_mask)
                ema_loss = self.DiceLoss(mask, ema_mask)
                
                self.log_dict({f"{stage} supervised loss": loss.item()}, prog_bar=True)
                

            # combined loss
            combinedLoss = loss + LAMBDA * loss_u
            combinedEmaLoss = ema_loss + LAMBDA * loss_u


        tp, fp, fn, tn = None, None, None, None
        if(not is_all_unlabeled):
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        if(stage == 'train'):
            self.ema.update()
            
        
        output = {
            "loss": combinedLoss,
            "ema_loss": combinedEmaLoss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        outputs.append(output)
        return output

    def shared_epoch_end(self, outputs, stage):

        # aggregate step metics
        tp = torch.cat([x['tp'] for x in outputs if x['tp'] != None])
        fp = torch.cat([x['fp'] for x in outputs if x['fp'] != None])
        fn = torch.cat([x['fn'] for x in outputs if x['fn'] != None])
        tn = torch.cat([x['tn'] for x in outputs if x['tn'] != None])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

        self.tb_log[f'{stage}_loss'] = np.mean([x['loss'].item() for x in outputs])
        self.tb_log[f'{stage}_ema_loss'] = np.mean([x['ema_loss'].item() for x in outputs])

        if(stage == 'train'):
            self.updateTB(self.tb_log, stage)
            self.epoch_counter +=1

    def training_step(self, batch, batch_idx):
        output = self.shared_step_fixmatch(batch, "train", self.inter_train_outputs)
        return output

    def on_train_epoch_end(self):
        return self.shared_epoch_end(self.inter_train_outputs, "train")

    def validation_step(self, batch, batch_idx):
        output = self.shared_step_fixmatch(batch, "valid", self.inter_val_outputs)
        return output

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.inter_val_outputs, "valid")

    def test_step(self, batch, batch_idx):
        output = self.shared_step_fixmatch(batch, "test", self.inter_test_outputs)
        return output

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.inter_test_outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)
    
    def updateTB(self, tb_log, stage):
        writer.add_scalars('DiceLoss', tb_log, global_step=self.epoch_counter)
        self.create_intermediate_steps()

    def create_intermediate_steps(self):
        self.inter_train_outputs = []
        self.inter_val_outputs = []
        self.inter_test_outputs = []

        self.tb_log = {}