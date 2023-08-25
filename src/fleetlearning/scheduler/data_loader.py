from zod import ZodFrames
from zod.constants import Anonymization
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import Generator
from fleetlearning.common.groundtruth_utils import get_ground_truth
from fleetlearning.common.static_params import global_configs
from fleetlearning.common.groundtruth_utils import load_ground_truth
from torch.utils.data import Dataset, RandomSampler


def load_datasets(partitioned_frame_ids: list, zod_frames: ZodFrames):
    seed = 42
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (global_configs.IMG_SIZE, global_configs.IMG_SIZE), antialias=True
            ),
        ]
    )

    trainset = ZodDataset(
        zod_frames=zod_frames,
        frames_id_set=partitioned_frame_ids,
        stored_ground_truth=load_ground_truth(global_configs.STORED_GROUND_TRUTH_PATH),
        transform=transform,
    )

    # Split each partition into train/val and create DataLoader
    len_val = int(len(trainset) * global_configs.VAL_FACTOR)
    len_train = int(len(trainset) - len_val)

    lengths = [len_train, len_val]
    ds_train, ds_val = random_split(trainset, lengths, Generator().manual_seed(seed))
    train_sampler = RandomSampler(ds_train)
    trainloader = DataLoader(
        ds_train,
        batch_size=global_configs.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        sampler=train_sampler,
    )
    valloader = DataLoader(ds_val, batch_size=global_configs.BATCH_SIZE, num_workers=4)

    # """report to tensor board"""
    # save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
    # save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

    return trainloader, valloader


class ZodDataset(Dataset):
    def __init__(
        self,
        zod_frames,
        frames_id_set,
        stored_ground_truth=None,
        transform=None,
        target_transform=None,
    ):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.target_transform = target_transform
        self.stored_ground_truth = stored_ground_truth

    def __len__(self):
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]
        image = frame.get_image(Anonymization.DNAT)
        label = None

        if self.stored_ground_truth:
            label = self.stored_ground_truth[frame_idx]
        else:
            label = get_ground_truth(self.zod_frames, frame_idx)

        label = label.astype("float32")
        image = image.astype("float32")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
