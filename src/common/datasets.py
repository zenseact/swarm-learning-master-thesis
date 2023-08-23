from common.static_params import global_configs
from common.groundtruth_utils import load_ground_truth, get_ground_truth
import torch
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from zod import ZodFrames
import zod.constants as constants
from zod.constants import Anonymization
from common.logger import fleet_log
from logging import INFO


class ZODImporter:
    def __init__(
        self,
        root=global_configs.DATASET_ROOT,
        subset_factor=global_configs.SUBSET_FACTOR,
        img_size=global_configs.IMG_SIZE,
        batch_size=global_configs.BATCH_SIZE,
        tb_path=None,
        stored_gt_path=None,
    ):
        version = "full"  # "mini" or "full"
        self.zod_frames = ZodFrames(dataset_root=root, version=version)

        validation_frames_all = self.zod_frames.get_split(constants.VAL)

        self.ground_truth = None
        if stored_gt_path:
            self.ground_truth = load_ground_truth(stored_gt_path)
            fleet_log(INFO, "loaded stored ground truth")

        validation_frames_all = [
            idx for idx in validation_frames_all if self.is_valid_frame(idx)
        ]

        self.validation_frames = validation_frames_all[
            : int(len(validation_frames_all) * subset_factor)
        ]

        fleet_log(INFO, f"test_frames length: {len(self.validation_frames)}")
        self.img_size = img_size
        self.batch_size = batch_size
        self.tb_path = tb_path

    def is_valid_frame(self, frame_id):
        if self.ground_truth:
            return frame_id in self.ground_truth
        else:
            return (
                get_ground_truth(self.zod_frames, frame_id).shape[0]
                == global_configs.OUTPUT_SIZE * 3
            )

    def load_test_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size), antialias=True),
            ]
        )

        testset = ZodDataset(
            zod_frames=self.zod_frames,
            frames_id_set=self.validation_frames,
            stored_ground_truth=self.ground_truth,
            transform=transform,
        )

        testloader = DataLoader(testset, batch_size=self.batch_size)

        # """report to tensor board"""
        # save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        # save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

        return testloader


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


def main(
    nr_clients=2,
    subset_factor=global_configs.SUBSET_FACTOR,
    img_size=global_configs.IMG_SIZE,
    batch_size=global_configs.BATCH_SIZE,
):
    # import Zod data into memory
    zod = ZODImporter(
        subset_factor=subset_factor,
        img_size=img_size,
        batch_size=batch_size,
        stored_gt_path=global_configs.STORED_GROUND_TRUTH_PATH,
    )

    # create pytorch loaders
    (
        trainloaders,
        valloaders,
        testloader,
        completeTrainloader,
        completeValloader,
    ) = zod.load_test_dataset()

    fleet_log(INFO, "nr of training imgs:", len(trainloaders[0].dataset))
    fleet_log(INFO, "nr of validation imgs:", len(valloaders[0].dataset))
    fleet_log(INFO, "nr of test imgs:", len(testloader.dataset))
    fleet_log(INFO, "input shape:", trainloaders[0].dataset[0][0].shape)
    fleet_log(INFO, "output shape:", trainloaders[0].dataset[0][1].shape)


if __name__ == "__main__":
    main()
