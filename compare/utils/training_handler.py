from torch.utils.tensorboard import SummaryWriter

from utils import ZODHandler as Data
from utils.training import run_centralized, run_federated


def run_training(config: dict, data: Data, log_dir=str) -> None:
    if config["central"]["train"] == "true":
        run_centralized(config, data, log_dir)
    if config["federated"]["train"] == "true":
        run_federated(
            config,
            data.federated.train.dataloaders,
            data.federated.val.dataloaders,
            data.test.dataloader,
            log_dir,
        )
