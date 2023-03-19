from collections import OrderedDict
import importlib
import logging
import numpy as np
import flwr as fl
import ray

from torch.nn import Module
from torch import Tensor, Size
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from ..models.model_manager import load_model
from .utils import train, test

logger = logging.getLogger(__name__)


def run_federated(
    config: dict,
    data: object,
    log_dir: str,
) -> None:
    logger.info("Initialising federated learning runtime")
    # Start the timer for federated training duration
    start_time = datetime.now()

    # Some parameters for federated learning
    n_clients = config["federated"]["global"]["n_clients"]
    global_rounds = config["federated"]["global"]["global_rounds"]
    client_resources = config["federated"]["global"]["client_resources"]
    ray_init_args = config["federated"]["global"]["ray_init_args"]

    # None-deep references to the dataloaders
    train_loaders = data.train.dataloaders
    val_loaders = data.val.dataloaders
    test_loader = data.test.dataloader

    # Load the loss function
    loss_method = config["model"]["loss"]
    module = importlib.import_module("torch.nn")
    loss_function = getattr(module, loss_method)

    # Evaluates the global model
    def global_evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        fl_config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Create a model and set the weights to the global model
        model = load_model(config)
        set_parameters(model, parameters)

        test_loss = test(
            network=model,
            dataloader=test_loader,
            loss_function=loss_function,
        )

        writer = SummaryWriter(log_dir)

        writer.add_scalar(
            "federated/loss/global",
            np.mean(test_loss),
            server_round,
        )

        writer.close()

    # The federated learning strategy

    def on_fit_config_fn(server_round: int):
        return dict(server_round=server_round)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        # initial_parameters=fl.common.ndarrays_to_parameters(server_params),
        evaluate_fn=global_evaluate,
        on_fit_config_fn=on_fit_config_fn,
    )

    # Define the client function
    def client_fn(cid: str) -> FlowerClient:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename="{}.log".format(Path(log_dir, "federated_{}".format(cid))),
            encoding="utf-8",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logger.debug("Creating client %s", cid)
        model = load_model(config)
        if config["federated"]["global"]["client_resources"]["num_gpus"] > 0:
            model.to("cuda")
        else:
            model.to("cpu")
        trainloader = train_loaders[int(cid)]
        valloader = val_loaders[int(cid)]
        client = FlowerClient(
            config,
            cid,
            model,
            trainloader,
            valloader,
            log_dir=log_dir,
        )
        return client

    # Start the federated learning simulation
    logger.debug("Starting federated learning simulation")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=global_rounds),
        ray_init_args=ray_init_args,
        client_resources=client_resources,
        strategy=strategy,
    )
    # Shutdown the ray cluster
    ray.shutdown()

    # Calculate the training duration
    end_time = datetime.now()
    soft_duration = str(end_time - start_time).split(".")[0]

    # Log that training is finished
    logger.info("Finished federated training after: {}".format(soft_duration))
    logger.info("END OF FEDERATED LEARNING")



# A method that fetches the model weights
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# The client class
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        config: dict,
        cid: int,
        model: Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        log_dir: str,
    ) -> None:
        self.config = config["federated"]["client"]
        self.cid = cid
        self.net = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = SummaryWriter(log_dir)

        # Load the loss function
        loss_method = config["model"]["loss"]
        module = importlib.import_module("torch.nn")
        self.loss_function = getattr(module, loss_method)

    def get_parameters(self, config):
        logger.debug("Get model parameters of client %s", self.cid)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        logger.debug("Start training of client %s", self.cid)
        set_parameters(self.net, parameters)
        logger.info("SERVER ROUND: {}".format(config["server_round"]))
        mean_epoch_loss, batch_losses, mean_validation_loss, network = train(
            network=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=self.config["epochs"],
            loss_function=self.loss_function,
            writer=self.writer,
            writer_path="federated/loss/clients/{}/".format(self.cid),
            server_round=config["server_round"],
        )
        new_parameters = get_parameters(self.net)
        return new_parameters, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        logger.debug("Start evaluation of client %s", self.cid)
        set_parameters(self.net, parameters)
        val_loss = test(
            network=self.net,
            dataloader=self.valloader,
            loss_function=self.loss_function,
        )

        return np.mean(val_loss), len(self.valloader), {}
