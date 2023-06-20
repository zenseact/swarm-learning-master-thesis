from collections import OrderedDict
import importlib
import logging
import numpy as np
import flwr as fl
import ray
import torch
import segmentation_models_pytorch as smp

from torch.nn import Module
from torch import Tensor, Size
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torchvision.transforms import *

from ..models.model_manager import load_model
from .utils import save_model, train, test, train_fixmatchseg

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
    
    # check if the model is fixmatchseg
    fixmatchseg = config["model"]["name"] == "fixmatchseg"

    # None-deep references to the dataloaders
    if not fixmatchseg:
        train_loaders = data.train.dataloaders
    else:
        labelled_train_loaders = data.train.labelled.dataloaders
        unlabelled_train_loaders = data.train.unlabelled.dataloaders  
    val_loaders = data.val.dataloaders
    full_val_loader = data.val.dataloader
    if len(data.test.dataset) > 0:
        test_loader = data.test.dataloader
    else:
        test_loader = None
        logger.debug("No test dataset provided, skipping test evaluation")

    # Load the loss function
    # Does not create an instance of the loss, just gets the class
    # flag for success
    success = False
    if not config["model"]["name"] == "fixmatchseg":
        try:
            loss_method = config["model"]["loss"]
            module = importlib.import_module("torch.nn")
            loss_class = getattr(module, loss_method)
            logger.debug("Using loss function: {}".format(loss_method))
            success = True
        except Exception as e:
            logger.error("Error loading loss function: {}".format(e))
            raise e
    else:
        loss_class = smp.losses.DiceLoss

    # Load the optimiser function
    try:
        opt_method = config["model"]["optimiser"]
        module = importlib.import_module("torch.optim")
        optimiser_class = getattr(module, opt_method)
        logger.debug("Using optimiser function: {}".format(optimiser_class))
    except KeyError:
        logger.debug("No optimiser specified, using Adam")
        optimiser_class = torch.optim.Adam
    except Exception as e:
        logger.error("Error loading loss function: {}".format(e))
        logger.exception(e)
        raise e

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
            dataloader=full_val_loader,
            loss_class=loss_class,
        )

        writer = SummaryWriter(log_dir)

        writer.add_scalar(
            "federated/loss/global",
            np.mean(test_loss),
            server_round,
        )

        # save the modellog_dir
        model_name = Path(
            log_dir, f"federated_{config['model']['name']}_{server_round}"
        )
        save_model(model, model_name)

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

        # Move the model to the correct device if needed
        if config["federated"]["global"]["client_resources"]["num_gpus"] > 0:
            model.to("cuda")
        else:
            model.to("cpu")
        
        valloader = full_val_loader  # val_loaders[int(cid)]

        if fixmatchseg:
            labelled_trainloader = labelled_train_loaders[int(cid)]
            unlablled_trainloader = unlabelled_train_loaders[int(cid)]
            trainloader = {"labelled": labelled_trainloader, "unlabelled": unlablled_trainloader}
            client = FlowerClient(
                config,
                cid,
                model,
                trainloader,
                valloader,
                log_dir=log_dir,
            )
        else:
            trainloader = train_loaders[int(cid)]
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
    state_dict = OrderedDict(
        {k: Tensor(v) if v.shape != Size([]) else Tensor([0]) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


# The client class
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        config: dict,
        cid: int,
        model: Module,
        trainloader: DataLoader|dict[DataLoader],
        valloader: DataLoader,
        log_dir: str,
    ) -> None:
        self.full_config = config
        self.config = config["federated"]
        self.cid = cid
        self.net = model
        if isinstance(trainloader, dict):
            self.fixmatchseg = True
            self.labelled_trainloader = trainloader.get("labelled")
            self.unlabelled_trainloader = trainloader.get("unlabelled")
        else:
            self.fixmatchseg = False
            self.trainloader = trainloader
        self.valloader = valloader
        self.writer = SummaryWriter(log_dir)

        # Load the loss function
        # Does not create an instance of the loss, just gets the class
        # flag for success
        success = False
        if not self.full_config["model"]["name"] == "fixmatchseg":
            try:
                loss_method = self.full_config["model"]["loss"]
                module = importlib.import_module("torch.nn")
                self.loss_class = getattr(module, loss_method)
                logger.debug("Using loss function: {}".format(loss_method))
                success = True
            except Exception as e:
                logger.error("Error loading loss function: {}".format(e))
                raise e
        else:
            self.loss_class = smp.losses.DiceLoss

        # Load the optimiser function
        try:
            opt_method = self.full_config["model"]["optimiser"]
            module = importlib.import_module("torch.optim")
            self.optimiser_class = getattr(module, opt_method)
            logger.debug("Using optimiser function: {}".format(self.optimiser_class))
        except KeyError:
            logger.debug("No optimiser specified, using Adam")
            self.optimiser_class = torch.optim.Adam
        except Exception as e:
            logger.error("Error loading loss function: {}".format(e))
            logger.exception(e)
            raise e

    def get_parameters(self, config):
        logger.debug("Get model parameters of client %s", self.cid)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Optimiser args
        try:
            optimiser_args = self.full_config["model"]["optimiser_args"]
        except KeyError:
            optimiser_args = {}

        logger.debug("Start training of client %s", self.cid)
        set_parameters(self.net, parameters)
        logger.info("SERVER ROUND: {}".format(config["server_round"]))
        if self.fixmatchseg:
            weak_augmentation = Compose(eval(self.full_config["data"]["weak_augmentation"]))
            strong_augmentation = Compose(eval(self.full_config["data"]["strong_augmentation"]))
            normalisation = Compose(eval(self.full_config["data"]["normalisation"]))
            mean_epoch_loss, batch_losses, mean_validation_loss, network = train_fixmatchseg(
                network=self.net,
                labelled_loader=self.labelled_trainloader,
                unlabelled_loader=self.unlabelled_trainloader,
                valloader=self.valloader,
                epochs=self.config["client"]["epochs"],
                loss_class=self.loss_class,
                optimiser_class=self.optimiser_class,
                weak_augmentation=weak_augmentation,
                strong_augmentation=strong_augmentation,
                normalisation=normalisation,
                optimiser_args=optimiser_args,
                writer=self.writer,
                writer_path="federated/loss/clients/{}/".format(self.cid),
                server_round=config["server_round"],
                tau=self.full_config["model"]["tau"],
            )
            new_parameters = get_parameters(self.net)
            return new_parameters, len(self.unlabelled_trainloader), {}

        else:
            mean_epoch_loss, batch_losses, mean_validation_loss, network = train(
                network=self.net,
                trainloader=self.trainloader,
                valloader=self.valloader,
                epochs=self.config["client"]["epochs"],
                loss_class=self.loss_class,
                optimiser_class=self.optimiser_class,
                optimiser_args=optimiser_args,
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
            loss_class=self.loss_class,
        )

        return np.mean(val_loss), len(self.valloader), {}
