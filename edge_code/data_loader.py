from zod import ZodFrames
from zod.constants import Anonymization
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import Generator
from common.groundtruth_utils import get_ground_truth
from common.static_params import *
from common.groundtruth_utils import load_ground_truth
from torch.utils.data import Dataset
import time


def load_datasets(partitioned_frame_ids: list):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME START = {current_time}")
        seed = 42
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))])
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME TRANSFORMS DONE = {current_time}")
        zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME ZOD FRAMES DONE= {current_time}")

        trainset = ZodDataset(zod_frames=zod_frames, frames_id_set=partitioned_frame_ids,
                              stored_ground_truth=load_ground_truth(STORED_GROUND_TRUTH_PATH), transform=transform)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME TRANSET FROM PARTITIONS DONE = {current_time}")
        testset = ZodDataset(zod_frames=zod_frames, frames_id_set=partitioned_frame_ids,
                             stored_ground_truth=load_ground_truth(STORED_GROUND_TRUTH_PATH), transform=transform)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME TESTSET FROM PARTITIONS DONE = {current_time}")

        # Split each partition into train/val and create DataLoader
        len_val = int(len(trainset) * VAL_FACTOR)
        len_train = int(len(trainset) - len_val)

        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(trainset, lengths, Generator().manual_seed(seed))
        trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME TRAINLOADER DONE = {current_time}")
        valloader = DataLoader(ds_val, batch_size=BATCH_SIZE)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME VALLOADER DONE = {current_time}")
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"TIME TESTLOADER DONE = {current_time}")

        # """report to tensor board"""
        # save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        # save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

        return trainloader, valloader, testloader


class ZodDataset(Dataset):
    def __init__(self, zod_frames, frames_id_set, stored_ground_truth=None, transform=None, target_transform=None):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform
        self.target_transform = target_transform
        self.stored_ground_truth = stored_ground_truth

    def __len__(self):
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]

        image = frame.get_image(Anonymization.DNAT)
        label = None

        if (self.stored_ground_truth):
            label = self.stored_ground_truth[frame_idx]
        else:
            label = get_ground_truth(self.zod_frames, frame_idx)

        label = label.astype('float32')
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label