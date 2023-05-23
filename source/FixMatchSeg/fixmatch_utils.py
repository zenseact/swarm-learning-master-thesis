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

def boundary_loss(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    boundary_pred = torch.abs(torch.gradient(pred)[0])
    boundary_target = torch.abs(torch.gradient(target)[0])

    intersection = (boundary_pred * boundary_target).sum()
    union = boundary_pred.sum() + boundary_target.sum()
    loss = 1 - ((2. * intersection + smooth) / (union + smooth))

    return loss.mean()


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    loss = (1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)))

    return loss.mean()


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
                dl = dice_loss(pred_strong[b].unsqueeze(0), pseudo_labels[b].unsqueeze(0)) # expand dimensions to match original dice_loss function requirements
                bl = boundary_loss(pred_strong[b].unsqueeze(0), pseudo_labels[b].unsqueeze(0)) # expand dimensions to match original boundary_loss function requirements
                loss += dl + bl
        loss = loss / mask_sum
    else:
        loss = torch.tensor(0.0, requires_grad=True).to(pred_weak.device)

    return loss