from randaugment import *
from static_params import *

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
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

