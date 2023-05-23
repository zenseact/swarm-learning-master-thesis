from randaugment import *
from static_params import *

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.05),
        ])
        
        self.strong = transforms.Compose([
            RandAugmentMC(n=2, m=10)])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(weak)
        return self.normalize(weak), self.normalize(strong)
    
    
def x_u_split(training_ids, validation_ids):
    labeled = random.sample(training_ids, k=int(len(training_ids)*LABELED_FACTOR))
    unlabeled = [x for x in training_ids if x not in labeled]
    print(len(labeled), 'are chosen as labeled and ', len(unlabeled), 'as unlabeled before augmentation')

    # make ids as tuple = (id, isLabeled)
    train_ids = [(x, True) for x in labeled]
    train_ids.extend([(x, False) for x in unlabeled])

    validation_ids =  [(x, True) for x in validation_ids]
    return train_ids, validation_ids

def boundary_loss(pred, target):
    loss = BoundaryLoss()
    pred = (pred.sigmoid() > 0.5).float()
    loss = loss(pred, target)
    return loss


def dice_loss(pred, target):
    loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    pred = (pred.sigmoid() > 0.5).float()
    loss = loss(pred, target)
    return loss

def compute_supervised_loss(preds, labels):
    # preds: Predicted segmentation masks by the model.
    # labels: Ground truth segmentation masks.
    B = preds.shape[0] # Get batch size

    # Initialize loss
    loss = 0

    # Loop over all images in the batch
    for b in range(B):
        # Compute Dice loss and Boundary loss for each image
        dl = dice_loss(preds[b], labels[b])
        bl = boundary_loss(preds[b], labels[b])
        # Add to total loss
        loss += dl + bl

    # Average the loss over the batch
    loss = loss / B

    return loss

def should_include_in_loss(pred_prob, threshold):
    max_confidences = pred_prob.squeeze(1) # [batch_size, height, width]
    avg_max_confidences = torch.mean(max_confidences, dim=[1, 2])
    mask = avg_max_confidences > threshold
    return mask

def compute_unsupervised_loss(pred_weak, pred_strong, threshold):
    pseudo_labels = (pred_weak > 0.5).float() # using 0.5 as a threshold to determine class
    mask = should_include_in_loss(pred_weak, threshold)
    
    loss = 0
    mask_sum = torch.sum(mask.float())
    
    if mask_sum.item() > 0:
        for b in range(mask.shape[0]):
            if mask[b]:
                dl = dice_loss(pred_strong[b], pseudo_labels[b]) # expand dimensions to match original dice_loss function requirements
                bl = boundary_loss(pred_strong[b], pseudo_labels[b]) # expand dimensions to match original boundary_loss function requirements
                loss += dl + bl
        loss = loss / mask_sum
    else:
        loss = torch.tensor(0.0, requires_grad=True).to(pred_weak.device)

    return loss

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before sigmoid)
                    shape (N, H, W)
            - gt: ground truth map
                    shape (N, H, W)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, _, _ = pred.shape

        # sigmoid so that predicted map can be distributed in [0, 1]
        pred = torch.sigmoid(pred)

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, -1)
        pred_b = pred_b.view(n, -1)
        gt_b_ext = gt_b_ext.view(n, -1)
        pred_b_ext = pred_b_ext.view(n, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=1) / (torch.sum(pred_b, dim=1) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=1) / (torch.sum(gt_b, dim=1) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss