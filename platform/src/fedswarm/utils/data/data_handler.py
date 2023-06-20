import importlib
import numpy as np
import logging
import matplotlib.pyplot as plt
import random

from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import *
from typing import List, Union
from zod.constants import *
from zod import ZodFrames
from torch.utils.tensorboard import SummaryWriter

from .extensions.custom_transforms import *

logger = logging.getLogger(__name__)


class DataObject:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        for key in kwargs:
            if key.startswith("_"):
                self.__create_property(key)

    def __create_property(self, name: str) -> None:
        def getter(self):
            return self.__dict__[name]

        setattr(self.__class__, name[1:], property(getter, None))

    def __len__(self) -> int:
        if hasattr(self, "dataset"):
            return len(self.dataset)
        else:
            raise AttributeError("No dataset attribute found")

    def create_dataloader(self) -> None:
        self._dataloader = DataLoader(self.dataset, **self._loader_args)

    def unmount_dataloader(self) -> None:
        logger.debug("Unmounting dataloader")
        del self._dataloader

    def create_dataloaders(self) -> None:
        self._dataloaders = [
            DataLoader(ds, **self._loader_args) for ds in self.datasets
        ]

    def unmount_dataloaders(self) -> None:
        logger.debug("Unmounting dataloaders")
        del self._dataloaders

    @property
    def dataloader(self) -> DataLoader:
        if not hasattr(self, "_dataset"):
            raise AttributeError("No dataset attribute found when creating dataloader")
        if len(self._dataset) == 0:
            logger.warning("Dataset is empty, skipping")
            return None
        if not hasattr(self, "_dataloader"):
            logger.warning(
                "Dataloader not created explicitly, creating it, do not forget to unmount it!"
            )
            self.create_dataloader()
        return self._dataloader

    @property
    def dataloaders(self) -> List[DataLoader]:
        if not hasattr(self, "_datasets"):
            raise AttributeError(
                "No datasets attribute found when creating dataloaders"
            )

        if not hasattr(self, "_dataloaders"):
            logger.warning(
                "Dataloaders not created explicitly, creating them, do not forget to unmount them!"
            )
            self.create_dataloaders()
        return self._dataloaders


class DataHandler:
    """
    The platform `DataHandler` object is a custom class that handles the dataloading and
    contains all the data required for training. The `DataHandler` is accesssed using the
    `data` attribute of the platform. The `DataHandler` always contains the following attributes:

        - `train` - the training data
        - `val` - the validation data
        - `test` - the test data

    The `train`, `val`, and `test` attributes are `utils.data.DataObject` objects.
    They are essentially just classes built from dictionaries so that they can be
    accessed using dot notation.

    Read more about the `DataHandler` class here: `src/example-1-daa.ipynb`

    """

    def __init__(self, full_config: dict, log_dir: str) -> None:
        logger.debug("Creating instance of DataHandler")
        # Add data specific config as config attribute
        self._config = full_config["data"]
        # Path to the data directory
        self._path: str = self._config["path"]
        # Evaluate the version string to a version constant
        self._version: str = eval((self._config["version"]).upper())
        logger.info("Loading ZodFrames...")
        self._zod_frames: ZodFrames = ZodFrames(self._path, self._version)
        logger.info("Finished loading ZodFrames")
        # Log directory for tensorboard
        self.log_dir = log_dir

        # Define the default train and validation frame id generator
        # Creates two lists of frame ids, one for the train set and one for the validation set
        def default_id_generator():
            logger.debug("Using default id generator")
            # Get the frame ids for the train and val sets
            self.__train_ids = self._zod_frames.get_split(TRAIN)
            self.__val_ids = self._zod_frames.get_split(VAL)

        # Check if a custom train_val_id_generator is defined in the config
        try:
            if self._config["train_val_id_generator"]:
                script_name = self._config["train_val_id_generator"]
                # Get the train_val_id_generator from the config
                generators = importlib.import_module(
                    f"fedswarm.utils.data.extensions.custom_datasets"
                )

                # Run the train_val_id_generator
                generator = getattr(generators, script_name)
                train_ids, val_ids = generator()

                # Set the train and val ids
                self.__train_ids = train_ids
                self.__val_ids = val_ids

                # Log that the custom train_val_id_generator is used
                logger.info("Using custom train_val_id_generator: %s" % script_name)
        except KeyError:
            logger.warning("No train_val_id_generator found, using default")
            default_id_generator()
        except Exception as e:
            logger.error("Could not use train_val_id_generator")
            logger.exception(e)
            logger.warning("Using default train_val_id_generator")
            default_id_generator()

        # Reduce the number of ids if specified in the config
        self.__train_ids = self._ratio(self._config["ratio"], self.__train_ids)
        if self._config.get("split_train") is not None:
            self.__train_ids_labelled = self._ratio(
                self._config["split_train"], self.__train_ids
            )
            self.__train_ids_unlabelled = self._ratio(
                self._config["split_train"], self.__train_ids, inverse=True
            )
        if not self._config.get("ratio_val"):
            self.__val_ids = self._ratio(self._config["ratio"], self.__val_ids)
        else:
            self.__val_ids = self._ratio(self._config["ratio_val"], self.__val_ids)
        # Shuffle the ids
        try:
            random.seed(self._config["shuffle_seed"])
            random.shuffle(self.__val_ids)
            if self._config.get("split_train") is not None:
                random.shuffle(self.__train_ids_labelled)
                random.shuffle(self.__train_ids_unlabelled)
            else:
                random.shuffle(self.__train_ids)
            logger.info("Shuffled ids with seed %s" % self._config["shuffle_seed"])
        except KeyError:
            logger.warning("No shuffle seed found, not shuffling ids")

        # Create a test set from the val set
        if not self._config.get("skip_test"):
            self.__test_ids = self.__val_ids[: len(self.__val_ids) // 2]
            self.__val_ids = self.__val_ids[len(self.__val_ids) // 2 :]
        else:
            self.__test_ids = []

        # The default transforms and additional transforms from config
        try:
            img_size = self._config["img_size"]
            transforms = [ToTensor()]
            try:
                result = eval(self._config["transforms"])
            except SyntaxError:
                result = []

            transforms += result

            logger.debug(
                "Added transforms from config: %s" % self._config["transforms"]
            )
            logger.debug("Transforms: %s" % transforms)

            # TODO: Add support via kwargs
            # TODO: Import transforms modules similar to how it is done in training_handler.py
        except Exception as e:
            logger.error(
                "Could not create transforms, using default [ToTensor()]. Consider aborting!"
            )
            logger.exception(e)
            transforms = [ToTensor()]

        # Creating the data objects for central training
        logger.info("Creating central datasets")

        self._test = DataObject(
            _ids=self.__test_ids,
            _dataset=ZodDataset(
                self._zod_frames,
                self.__test_ids,
                transforms=transforms,
                config=full_config,
            ),
            _loader_args=self._config["dataloader_args"],
        )
            
        if self._config.get("split_train") is not None:
            logger.debug("Splitting train set into labelled and unlabelled")
            # parse batch size argument, if a list is given, use first argument for labelled and second for unlabelled
            if type(self._config["dataloader_args"]["batch_size"]) is list:
                logger.debug("Batch size is list, using first argument for labelled and second for unlabelled")
                l_s, u_s = self._config["dataloader_args"]["batch_size"]
                data_loader_args_labelled = self._config["dataloader_args"].copy()
                data_loader_args_labelled.update({"batch_size": l_s})
                data_loader_args_unlabelled = self._config["dataloader_args"].copy()
                data_loader_args_unlabelled.update({"batch_size": u_s})
            
            self._train_labelled = DataObject(
                _ids=self.__train_ids_labelled,
                _dataset=ZodDataset(
                    self._zod_frames,
                    self.__train_ids_labelled,
                    transforms=transforms,
                    config=full_config,
                ),
                _loader_args=data_loader_args_labelled,
            )
            if data_loader_args_unlabelled["batch_size"] == 0:
                logger.debug("Unlabelled batch size is 0, skipping unlabelled dataset")
                self._train_unlabelled = []
            else:
                self._train_unlabelled = DataObject(
                    _ids=self.__train_ids_unlabelled,
                    _dataset=ZodDataset(
                        self._zod_frames,
                        self.__train_ids_unlabelled,
                        transforms=transforms,
                        config=full_config,
                    ),
                    _loader_args=data_loader_args_unlabelled,
                )
            logger.debug("Adding labelled and unlabelled datasets to train data object")
            self._train = DataObject(
                labelled=self._train_labelled, unlabelled=self._train_unlabelled
            )
            assert hasattr(self._train, "labelled"), "Train object does not have labelled attribute"
            assert hasattr(self._train, "unlabelled"), "Train object does not have unlabelled attribute"
        else:
            logger.debug("Not splitting train set into labelled and unlabelled")
            self._train = DataObject(
                _ids=self.__train_ids,
                _dataset=ZodDataset(
                    self._zod_frames,
                    self.__train_ids,
                    transforms=transforms,
                    config=full_config,
                ),
                _loader_args=self._config["dataloader_args"],
            )
        if self._config.get("split_train") is not None:
            self._config["dataloader_args"]["batch_size"] = sum(self._config["dataloader_args"]["batch_size"])            
        self._val = DataObject(
            _ids=self.__val_ids,
            _dataset=ZodDataset(
                self._zod_frames,
                self.__val_ids,
                transforms=transforms,
                config=full_config,
            ),
            _loader_args=self._config["dataloader_args"],
        )

        # Set the central data objects as attributes
        setattr(
            self,
            "central",
            DataObject(_train=self._train, _val=self._val, _test=self._test),
        )

        # Log the number of samples in each set
        if self._config.get("split_train") is not None: 
            logger.info(
                "Training samples: (labelled: {}, unlabelled: {}), Validation samples: {}, Test samples: {} ".format(
                    len(self._train_labelled),
                    len(self._train_unlabelled),
                    len(self._val),
                    len(self._test),
                )
            )
        else:
            logger.info(
                "Training samples: {}, Validation samples: {}, Test samples: {}".format(
                    len(self._train), len(self._val), len(self._test)
                )
            )

        # Create decentralised data objects
        if full_config["federated"]["train"]:
            self.create_decentralised_datasets(full_config, "federated")
        if full_config["swarm"]["train"]:
            self.create_decentralised_datasets(full_config, "swarm")
        if full_config["baseline"]["train"]:
            self.create_decentralised_datasets(full_config, "baseline")

        logger.info("DataHandler initialisation complete")

    def create_decentralised_datasets(self, config: dict, method: str) -> None:
        n_clients = config[method]["global"]["n_clients"]
        logger.info("[{}] Splitting dataset for {} clients".format(method, n_clients))

        writer = SummaryWriter(self.log_dir)
        if config["data"].get("split_train"):
            train_datasets_labelled = [self._train.labelled.dataset] * n_clients
            sample_distribution = [len(x) for x in train_datasets_labelled]
            logger.debug("Sample distribution: {}".format(sample_distribution))
            # legacy code below, not sure it is needed.
            # split_dataset(
            #     self._train.labelled.dataset,
            #     n_clients,
            #     seed=self._config["shuffle_seed"],
            #     distribution="uniform",
            #     writer=writer,
            #     subtitle="training_labelled",
            # )
            train_datasets_unlabelled = split_dataset(
                self._train.unlabelled.dataset,
                n_clients,
                seed=self._config["shuffle_seed"],
                distribution="uniform",
                writer=writer,
                subtitle="training_unlabelled",
            )
            train = DataObject(
                labelled = DataObject(
                    _datasets=train_datasets_labelled,
                    _loader_args=self._config["dataloader_args"],
                ), 
                unlabelled = DataObject(
                    _datasets=train_datasets_unlabelled,
                    _loader_args=self._config["dataloader_args"],
                )
            )
            
        else:
            train_datasets = split_dataset(
                self._train.dataset,
                n_clients,
                seed=self._config["shuffle_seed"],
                distribution="uniform",
                writer=writer,
                subtitle="training",
            )
            train = DataObject(_datasets=train_datasets, _loader_args=self._config["dataloader_args"])

        validation_datasets = split_dataset(
            self._val.dataset,
            n_clients,
            seed=self._config["shuffle_seed"],
            distribution="uniform",
            writer=writer,
            subtitle="validation",
        )
        setattr(
            self,
            method,
            DataObject(
                _n_clients=n_clients,
                _train=train,
                _val=DataObject(
                    _dataset=self._val.dataset,
                    _datasets=validation_datasets,
                    _loader_args=self._config["dataloader_args"],
                ),
                _test=self._test,
            ),
        )

        writer.close()

    def _ratio(self, ratio, data, inverse=False):
        if inverse:
            return data[int(len(data) * ratio) :]
        else:
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
            logger.ERROR("Test set not created")
        return self._test


class ZodDataset(Dataset):
    def __init__(
        self,
        zod_frames,
        frames_id_set,
        config,
        transforms=None,
        target_transform=None,
    ) -> None:
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transforms = Compose(transforms)
        self.target_transform = target_transform
        self.config = config

        # Check if a custom get_method is defined in the config
        try:
            if self.config["data"]["dataset_getitem"]:
                # get the method name from the config
                dataset_getitem_method_name = self.config["data"]["dataset_getitem"]
                # get the dataset get method from the custom_dataset.py file
                custom_methods = importlib.import_module(
                    f"fedswarm.utils.data.extensions.custom_datasets"
                )
                dataset_getitem_method = getattr(
                    custom_methods, dataset_getitem_method_name
                )
                self.get_method = dataset_getitem_method
                # Log that the custom train_val_id_generator is used
                logger.info(
                    "Using custom dataset get method: %s" % dataset_getitem_method_name
                )
        except KeyError as e:
            logger.warning("No dataset_getitem method found")
            raise e
        except Exception as e:
            logger.error("Could not use dataset_getitem method")
            raise e

    def __len__(self) -> int:
        return len(self.frames_id_set)

    def __getitem__(self, idx: int):
        label, image = self.get_method(self, idx)
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
    """
    Split dataset into `num_clients` partitions
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    if type(distribution) is str:
        if distribution == "random":
            logger.error("Random distribution not implemented")
            raise NotImplementedError
        elif distribution == "uniform":
            partition_size = len(data) // n
            lengths = [partition_size] * (n - 1)
            lengths.append(len(data) - sum(lengths))
        else:
            logger.error("Distribution method not implemented")
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
            logger.warning("Distribution error greater than 5%")
            raise ValueError

    datasets = random_split(data, lengths, Generator().manual_seed(seed))

    sample_distribution = [len(x) for x in datasets]
    logger.debug("Sample distribution: {}".format(sample_distribution))

    # Plot histogram of sample distribution
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    plt.bar(list(range(1, len(sample_distribution) + 1)), sample_distribution)
    plt.xlabel("Partitions")
    plt.ylabel("Samples")
    plt.suptitle("Distribution of samples")
    plt.title("{}, seed: {}".format(subtitle, seed))
    try:
        writer.add_figure(
            "sample_distribution/{}".format(subtitle), plt.gcf(), global_step=0
        )
        logger.info(
            "[TENSORBOARD] Added decentralised client data distribution figure to tensorboard"
        )
    except Exception as e:
        logger.exception(
            "[TENSORBOARD] Could not add figure to tensorboard: {}".format(e)
        )
    return datasets
