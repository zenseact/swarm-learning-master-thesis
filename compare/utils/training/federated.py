from collections import OrderedDict
import importlib
import logging
import numpy as np
import flwr as fl

from torch.nn import Module
from torch import Tensor, Size
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from utils import load_model, ZODHandler as Data
from utils.training import train, test


def run_federated(
    config: dict,
    training_loaders: List[DataLoader],
    validation_loaders: List[DataLoader],
    test_loader: DataLoader,
    log_dir: str,
) -> None:
    # Some parameters for federated learning
    n_clients = config["federated"]["global"]["n_clients"]
    global_rounds = config["federated"]["global"]["global_rounds"]
    client_resources = config["federated"]["global"]["client_resources"]

    # Load the loss function
    loss_method = config["central"]["loss"]
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

        writer.add_scalars(
            "federated/loss",
            {"global": np.mean(test_loss)},
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
        model = load_model(config)
        if config["federated"]["global"]["client_resources"]["num_gpus"] > 0:
            model.to("cuda")
        else:
            model.to("cpu")
        trainloader = training_loaders[int(cid)]
        valloader = validation_loaders[int(cid)]
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
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=global_rounds),
        client_resources=client_resources,
        strategy=strategy,
    )


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
        loss_method = config["central"]["loss"]
        module = importlib.import_module("torch.nn")
        self.loss_function = getattr(module, loss_method)

    def get_parameters(self, config):
        logging.debug("Get model parameters of client %s", self.cid)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        logging.debug("Start training of client %s", self.cid)
        set_parameters(self.net, parameters)
        print("SERVER ROUND", config["server_round"])
        epoch_loss, batch_losses = train(
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
        logging.debug("Start evaluation of client %s", self.cid)
        set_parameters(self.net, parameters)
        val_loss = test(
            network=self.net,
            dataloader=self.valloader,
            loss_function=self.loss_function,
        )

        return np.mean(val_loss), len(self.valloader), {}
