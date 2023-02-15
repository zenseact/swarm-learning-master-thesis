import numpy as np
import logging
import matplotlib.pyplot as plt

from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import *
from typing import List, Union
from zod.constants import *
from zod import ZodFrames
from torch.utils.tensorboard import SummaryWriter


class DataObject:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        for key in kwargs:
            if key.startswith("_"):
                self.__create_property(key)

        # Create a dataloader if a dataset and batch size is given
        if all(x in kwargs for x in ["_dataset", "_batch_size"]):
            self.dataloader = DataLoader(self.dataset, self.batch_size)

        # Create a list of dataloaders if a list of datasets and batch size is given
        if all(x in kwargs for x in ["_datasets", "_batch_size"]):
            self.dataloaders = [DataLoader(ds, self.batch_size) for ds in self.datasets]

    def __create_property(self, name: str) -> None:
        def getter(self):
            return self.__dict__[name]

        setattr(self.__class__, name[1:], property(getter, None))

    def __len__(self) -> int:
        if hasattr(self, "dataset"):
            return len(self.dataset)
        else:
            raise AttributeError("No dataset attribute found")


class ZODHandler:
    def __init__(self, full_config: dict, log_dir: str) -> None:
        self._config = full_config["data"]
        self._path: str = "/staging/dataset_donation/round_2"
        self._version: str = eval((self._config["version"]).upper())
        self._zod_frames: ZodFrames = ZodFrames(self._path, self._version)

        self.__train_ids = self._ratio(
            self._config["ratio"], self._zod_frames.get_split(TRAIN)
        )

        self.__val_ids = self._ratio(
            self._config["ratio"], self._zod_frames.get_split(VAL)
        )

        # The default transforms
        img_size = self._config["img_size"]
        transforms = [ToTensor(), Resize((img_size, img_size))]

        # Add any additional transforms specified in the config
        # TODO: Add support via kwargs
        # TODO: Import transforms modules similar to how it is done in training_handler.py
        transforms += eval(self._config["transforms"])

        # Creating the data objects
        if self._config["test_set"] == "true":
            self.__test_ids = self.__val_ids[: len(self.__val_ids) // 2]
            self.__val_ids = self.__val_ids[len(self.__val_ids) // 2 :]

            self._test = DataObject(
                _ids=self.__test_ids,
                _dataset=ZodDataset(
                    self._zod_frames, self.__test_ids, transforms=transforms
                ),
                _batch_size=self._config["batch_size"],
            )
        else:
            self._test = None

        self._train = DataObject(
            _ids=self.__train_ids,
            _dataset=ZodDataset(
                self._zod_frames, self.__train_ids, transforms=transforms
            ),
            _batch_size=self._config["batch_size"],
        )

        self._val = DataObject(
            _ids=self.__val_ids,
            _dataset=ZodDataset(
                self._zod_frames, self.__val_ids, transforms=transforms
            ),
            _batch_size=self._config["batch_size"],
        )

        # Log the number of samples in each set
        logging.info(
            "Training samples: %s, Validation samples: %s, Test samples: %s"
            % (len(self._train), len(self._val), len(self._test))
        )

        if full_config["federated"]["train"] == "true":
            n_clients = full_config["federated"]["global"]["n_clients"]

            logging.info("Splitting dataset for %s clients" % n_clients)

            writer = SummaryWriter(log_dir)

            train_datasets = split_dataset(
                self._train.dataset,
                n_clients,
                distribution="uniform",
                writer=writer,
                subtitle="training",
            )
            validation_datasets = split_dataset(
                self._val.dataset,
                n_clients,
                distribution="uniform",
                writer=writer,
                subtitle="validation",
            )
            self.federated = DataObject(
                _n_clients=n_clients,
                _train=DataObject(
                    _datasets=train_datasets, _batch_size=self._config["batch_size"]
                ),
                _val=DataObject(
                    _datasets=validation_datasets,
                    _batch_size=self._config["batch_size"],
                ),
            )

            writer.close()

    def _ratio(self, ratio, data):
        # TODO: shuffle data? and hash indicies?
        return data[: int(len(data) * ratio)]

    @property
    def config(self) -> dict:
        return self._config

    @property
    def path(self) -> str:
        return self._path

    @property
    def version(self) -> str:
        return self._version

    @property
    def zod_frames(self) -> ZodFrames:
        return self._zod_frames

    @property
    def train(self) -> DataObject:
        return self._train

    @property
    def val(self) -> DataObject:
        return self._val

    @property
    def test(self) -> DataObject:
        if self._test is None:
            logging.ERROR("Test set not created")
        return self._test


class ZodDataset(Dataset):
    def __init__(
        self, zod_frames, frames_id_set, transforms=None, target_transform=None
    ) -> None:
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transforms = Compose(transforms)
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]

        image = frame.get_image(Anonymization.DNAT)
        poses = frame.ego_motion.poses
        x = poses[:, 0:1, 3]
        y = poses[:, 1:2, 3]
        z = poses[:, 2:3, 3]
        coordinates = np.append(x, y)
        coordinates = np.append(coordinates, z)

        label = coordinates.astype("float32")
        image = image.astype("float32")

        if self.transforms:
            image = self.transforms(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def split_dataset(
    data: Dataset,
    n: int,
    seed: int = None,
    distribution: Union[List, str] = "uniform",
    writer: SummaryWriter = None,
    subtitle: str = None,
) -> List[Dataset]:
    """ "
    Split dataset into `num_clients` partitions
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    if type(distribution) is str:
        if distribution == "random":
            logging.ERROR("Random distribution not implemented")
            raise NotImplementedError
        elif distribution == "uniform":
            partition_size = len(data) // n
            lengths = [partition_size] * (n - 1)
            lengths.append(len(data) - sum(lengths))
        else:
            logging.ERROR("Distribution method not implemented")
            raise NotImplementedError
    elif type(distribution) is List:
        # check valid distribution
        assert sum(distribution) == 1.0, "Distribution must sum to 1"
        # transform distribution to lengths
        lengths = [int(len(data) * d) for d in distribution]
        # check error size
        pct_errors = [1 - p / len(data) / d for p, d in zip(lengths, distribution)]
        # if error is greater than 5%, raise error
        if max(pct_errors) > 0.05:
            logging.ERROR("Distribution error greater than 5%")
            raise ValueError

    datasets = random_split(data, lengths, Generator().manual_seed(seed))

    sample_distribution = [len(x) for x in datasets]
    logging.debug("Sample distribution: %s" % sample_distribution)
    # Plot histogram of sample distribution
    plt.bar(list(range(1, len(sample_distribution) + 1)), sample_distribution)
    plt.xlabel("Partitions")
    plt.ylabel("Samples")
    plt.suptitle("Distribution of samples")
    plt.title("%s, seed: %s" % (subtitle, seed))
    writer.add_figure("sample_distribution/%s" % (subtitle), plt.gcf(), global_step=0)
    return datasets
