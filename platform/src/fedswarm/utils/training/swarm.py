from __future__ import annotations
from collections import OrderedDict
import importlib
import os

import ray
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import logging
import torch
import segmentation_models_pytorch as smp

from uuid import uuid4
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple
from torch.nn import Module
from copy import deepcopy
from torch import Tensor
from pathlib import Path
from ray.actor import ActorHandle

from ..models.model_manager import load_model
from .utils import train, get_best_device, train_fixmatchseg
from ..data.data_handler import DataObject
from .aggregators import average

from torchvision.transforms import *

logger = logging.getLogger(__name__)


def run_swarm(
    config: dict,
    data: DataObject,
    log_dir: str,
    baseline: bool = False,
) -> None:
    try:
        config_key = "baseline" if baseline else "swarm"
        logger.info("Initialising Ray runtime")

        ray.init(**config[config_key]["global"]["ray_init_args"])
    except Exception as e:
        logger.error("Error initialising Ray runtime: {}".format(e))
        logger.exception(e)
        raise e
    try:
        logger.info("Initialising swarm learning runtime")
        if baseline:
            logger.info("Initialising swarm learning with baseline parameters")
        sim = SimulationRunner(
            config=config,
            log_dir=log_dir,
            data=data,
            aggregation_function=average,
            baseline=baseline,
        )
        logger.info("Swarm learning runtime initialised")
    except Exception as e:
        logger.error("Error initialising swarm learning runtime: {}".format(e))
        raise e

    sim.spawn_clients()
    sim.start()
    ray.shutdown()
    logger.info("END OF SWARM LEARNING")


class TemporalModel:
    # A class which saves the parameters of a model and the time it was created
    # Every temporal model has a unique id (mid)
    def __init__(self, cid: str, mid: str, torch_model: Module) -> None:
        self.mid = mid
        self.model = torch_model
        self.created_by = cid
        self.created_at = datetime.now()

    def get_parameters(self) -> Dict[str, Tensor]:
        return self.model.state_dict()


@ray.remote
class Client:
    def __init__(
        self,
        cid,
        icid,
        trainloader,
        valloader,
        epochs,
        models: object,
        loss_class,
        optimiser_class,
        torch_model: Module,
        log_dir: str,
        method_string: str,
        config: dict,
        weak_augmentation=None,
        strong_augmentation=None,
        normalisation=None,
    ) -> None:
        self.full_config = config
        self.cid = cid
        self.icid = icid
        if isinstance(trainloader, dict):
            self.fixmatchseg = True
            self.labelled_trainloader = trainloader.get("labelled")
            self.unlabelled_trainloader = trainloader.get("unlabelled")
            self.weak_augmentation = weak_augmentation
            self.strong_augmentation = strong_augmentation
            self.normalisation = normalisation
        else:
            self.fixmatchseg = False
            self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.model = TemporalModel(id, uuid4().hex, torch_model)
        self.loss_class = loss_class
        self.optimiser_class = optimiser_class
        self.models = models
        self.aggregation_queue = []
        self.log_dir = log_dir
        self.round = 0
        self.method_string = method_string

        # Logging
        logging.basicConfig(
            filename="{}_{}.log".format(
                Path(self.log_dir, self.method_string), self.cid
            ),
            encoding="utf-8",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

        logger.debug("Visible: {}".format(os.environ.get("CUDA_VISIBLE_DEVICES", None)))
        self.model.model = self.model.model.cuda()

    # Get and set methods

    def get_cid(self) -> str:
        return self.cid

    def get_model(self) -> TemporalModel:
        return self.model

    def get_models(self) -> Dict[str, TemporalModel]:
        return self.models

    def get_aggregation_queue(self) -> List[TemporalModel]:
        return self.aggregation_queue

    # Simulation coordination methods

    def push_to_history(self) -> str:
        logger.debug(
            "{cid} Adding model {mid} to history".format(
                cid=self.cid[:7], mid=self.model.mid[:7]
            )
        )
        self.model.model.cpu()
        # Adds the current temporal model to the history
        self.models.update({self.model.mid: self.model})
        return self.cid, self.model.mid, self.models

    def add_to_aggregation_queue(self, model: TemporalModel) -> None:
        self.aggregation_queue.append(model)
        logger.debug(
            "{cid} Added model {mid} to aggregation queue".format(
                cid=self.cid[:7], mid=model.mid[:7]
            )
        )

    def set_model(self, model_weights: OrderedDict) -> None:
        model = deepcopy(self.model.model)
        model.load_state_dict(model_weights, strict=True)
        model.cuda()
        new_model_mid = uuid4().hex
        logger.debug(
            "{cid} Setting new model {mid}".format(
                cid=self.cid[:7], mid=new_model_mid[:7]
            )
        )
        self.model = TemporalModel(self.cid, new_model_mid, model)

    def fit(self) -> None:
        self.round += 1
        logger.info(
            "{} Starting training process on model {}".format(
                self.cid[:7], self.model.mid[:7]
            )
        )

        # Optimiser args
        try:
            optimiser_args = self.full_config["model"]["optimiser_args"]
        except KeyError:
            optimiser_args = {}
            
        if self.fixmatchseg:
            train_fixmatchseg(
                network=self.model.model,
                labelled_loader=self.labelled_trainloader,
                unlabelled_loader=self.unlabelled_trainloader,
                valloader=self.valloader,
                epochs=self.epochs,
                loss_class=self.loss_class,
                optimiser_class=self.optimiser_class,
                weak_augmentation=self.weak_augmentation,
                strong_augmentation=self.strong_augmentation,
                normalisation=self.normalisation,
                optimiser_args=optimiser_args,
                writer=SummaryWriter(self.log_dir),
                writer_path="{}/loss/clients/{}/".format(self.method_string, self.icid),
                server_round=self.round,
                tau=self.full_config["model"]["tau"],
            )
        else:
            train(
                network=self.model.model,
                trainloader=self.trainloader,
                valloader=self.valloader,
                epochs=self.epochs,
                loss_class=self.loss_class,
                optimiser_class=self.optimiser_class,
                optimiser_args=optimiser_args,
                writer=SummaryWriter(self.log_dir),
                writer_path="{}/loss/clients/{}/".format(self.method_string, self.icid),
                server_round=self.round,
            )
        logger.info("{} Training finished".format(self.cid[:7]))


class SimulationRunner:
    def __init__(
        self,
        config: dict,
        data: DataObject,
        log_dir: str,
        aggregation_function: callable,
        baseline: bool = False,
    ) -> None:
        self.config = config
        self.topology_round = 0
        self.baseline = baseline
        self.fixmatchseg = config["model"]["name"] == "fixmatchseg"

        # None-deep references to the dataloaders
        if not self.fixmatchseg:
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
            
        # Swarm interpretation of datasets
        train_loaders = train_loaders if not self.fixmatchseg else labelled_train_loaders
        
        # Load augmentation and normalisation if fixmatchseg
        if self.fixmatchseg:
            self.weak_augmentation = Compose(eval(self.config["data"]["weak_augmentation"]))
            self.strong_augmentation = Compose(eval(self.config["data"]["strong_augmentation"]))
            self.normalisation = Compose(eval(self.config["data"]["normalisation"]))
            self.fixmatch_payload = {"weak_augmentation": self.weak_augmentation,
                                    "strong_augmentation": self.strong_augmentation,
                                    "normalisation": self.normalisation,
                                    "train_loaders_unlabelled": unlabelled_train_loaders}
        # Init the simulator
        logger.debug("Creating simulator instance...")
        self.sim = Simulator.remote(
            config=config,
            log_dir=log_dir,
            train_loaders=train_loaders,
            val_loaders=full_val_loader,
            test_loader=test_loader,
            aggregation_function=aggregation_function,
            baseline=baseline,
            fixmatch_payload=self.fixmatch_payload if self.fixmatchseg else None,
        )
        logger.debug("Simulator instance created")

    def spawn_clients(self) -> None:
        # Concurrently spawn the clients
        try:
            logger.info("Spawning clients in ray workers...")
            logger.info("More information in the ray logs [swarm.log]")
            ray.get(self.sim.spawn_clients.remote())
            logger.debug("Clients spawned")
        except Exception as e:
            logger.error("Error spawning clients: {}".format(e))
            raise e

    def set_topology(self, topology_generator: callable) -> None:
        # Increment the topology number
        logger.debug("Topology round: {}".format(self.topology_round))
        self.topology_round += 1

        # Concurrently set the topology
        try:
            logger.info("Setting topology in ray workers...")
            logger.info("More information in the ray logs [swarm.log]")
            ray.get(self.sim.set_topology.remote(topology_generator))
            logger.debug("Topology set")
        except Exception as e:
            logger.error("Error setting topology: {}".format(e))
            raise e

    def propagate_topology(self) -> None:
        # Concurrently propagate the topology
        try:
            logger.info("Propagating topology in ray workers...")
            logger.info("More information in the ray logs [swarm.log]")
            intents = ray.get(self.sim.send_intent.remote())
            ray.get(self.sim.send_jobs.remote(intents))
            logger.debug("Topology propagated")
        except Exception as e:
            logger.error("Error propagating topology: {}".format(e))
            raise e

        try:
            self.sim.plot_network.remote(self.topology_round)
        except Exception as e:
            logger.error("Error plotting network: {}".format(e))
            raise e

    def train(self) -> None:
        # Concurrently train the clients
        try:
            logger.info("Training clients in ray workers...")
            logger.info("More information in the ray logs [swarm.log]")
            ray.get(self.sim.train.remote())
            logger.debug("Clients trained")
        except Exception as e:
            logger.error("Error training clients: {}".format(e))
            raise e

    def start_communication(self) -> None:
        logger.info("Starting aggregation process in ray workers...")
        logger.info("More information in the ray logs [swarm.log]")
        # Get the clients and init remote futures
        clients = ray.get(self.sim.get_clients.remote())
        futures = [self.sim.aggregate.remote(c) for c in clients.values()]
        logger.debug("Created aggregation futures")
        # Run the aggregation process in parallel
        ray.get(futures)
        logger.debug("Aggregation process finished")

    # Strategy start callable
    def start(self) -> None:
        logger.info("Starting simulation loop")
        module = importlib.import_module("fedswarm.utils.training.strategies")

        # load swarm orchestrator
        if self.baseline:
            strategy = getattr(module, "synchronous_fixed_rounds_edgeless")
        else:
            try:
                name = self.config["swarm"]["global"]["orchestrator"]
                strategy = getattr(module, name)
            except KeyError:
                logger.warning("No orchestrator specified, using default")
                strategy = getattr(module, "synchronous_fixed_rounds_fc")
            except AttributeError:
                logger.error("Orchestrator '{}' not found".format(name))
                raise AttributeError("Orchestrator '{}' not found".format(name))

        # run the strategy
        strategy(self)


@ray.remote
class Simulator:

    def __init__(
        self,
        config: dict,
        train_loaders: List[DataObject],
        val_loaders,
        test_loader: DataObject,
        log_dir: str,
        aggregation_function: callable,
        baseline: bool = False,
        fixmatch_payload: dict = None,

    ) -> None:
        # Shared variables
        self.clients: Dict[str, ActorHandle[Client]] = {}
        self.models: Dict[str, TemporalModel] = {}

        # Swarm or baseline string variable
        self._method_string = "baseline" if baseline else "swarm"
        
        # Sim properties
        self._n_clients = config[self._method_string]["global"]["n_clients"]
        self._topology: List[str] = []
        self._aggregation_function = aggregation_function
        self._config = config
        self._log_dir = log_dir
        self._train_loaders = train_loaders
        self._val_loaders = val_loaders
        self._test_loader = test_loader
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self._client_resources = config[self._method_string]["global"][
            "client_resources"
        ]
        if fixmatch_payload is not None:
            self.fixmatchseg = True
            self.fixmatch_payload = fixmatch_payload
            
        # Logging
        logger = logging.getLogger(__name__)

        # Load the loss function
        # Does not create an instance of the loss, just gets the class
        # flag for success
        success = False
        if not self.fixmatchseg:
            try:
                loss_method = self._config["model"]["loss"]
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
            opt_method = self._config["model"]["optimiser"]
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

        # Load the optimiser function
        try:
            opt_method = self._config["model"]["optimiser"]
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

        logging.basicConfig(
            filename="{}.log".format(Path(self._log_dir, self._method_string)),
            encoding="utf-8",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    # Getter and setter methods
    def get_clients(self):
        return self.clients

    def get_models(self):
        return self.models

    def get_topology(self):
        return self._topology

    def set_topology(self, topology_generator: callable):
        self._topology = topology_generator(list(self.clients.keys()))

    # Support methods
    def create_client_actor(self, index: int) -> Tuple[str, Client]:
        id = uuid4().hex
        self.models[id] = {}
        model = load_model(self._config)

        logger.info("Created client with short id: {}".format(id[:7]))
        if isinstance(self._val_loaders, list):
            val_loader = self._val_loaders[index]
        else:
            val_loader = self._val_loaders
        
        if self.fixmatchseg:
            labelled_trainloader = self._train_loaders[int(index)]
            unlablled_trainloader = self.fixmatch_payload["train_loaders_unlabelled"][int(index)]
            trainloader = {"labelled": labelled_trainloader, "unlabelled": unlablled_trainloader}
            
            client_ref = Client.options(**self._client_resources).remote(
                config=self._config,
                cid=id,
                icid=index,
                trainloader=trainloader,
                valloader=val_loader,
                epochs=self._config[self._method_string]["client"]["epochs"],
                models=self.models[id],
                loss_class=self.loss_class,
                optimiser_class=self.optimiser_class,
                torch_model=model,
                log_dir=self._log_dir,
                method_string=self._method_string,
                weak_augmentation=self.fixmatch_payload["weak_augmentation"],
                strong_augmentation=self.fixmatch_payload["strong_augmentation"],
                normalisation=self.fixmatch_payload["normalisation"],
            )
        return id, client_ref

    def spawn_clients(self) -> None:
        try:
            logger.info("Spawning client actors...")
            self.clients.update(
                [self.create_client_actor(i) for i in range(self._n_clients)]
            )
            logger.info("Client actors spawned")
        except Exception as e:
            logger.error("Error while spawning clients: {}".format(e))
            raise e

    def send_intent(self) -> List[object]:
        try:
            logger.debug("Sending intents...")
            futures = [
                client_actor.push_to_history.remote()
                for client_actor in self.clients.values()
            ]
            logger.debug("Futures for sending intents created")
            results = ray.get(futures)

            # Update the sim model object from the actors
            for cid, mid, models in results:
                self.models[cid].update(models)

            references = {cid: mid for cid, mid, _ in results}
            logger.debug("Intents sent")
            return references

        except Exception as e:
            logger.error("Error while sending intents: {}".format(e))
            raise e

    def send_jobs(self, references: dict) -> None:
        logger.debug("Sending jobs...")
        for cid_s, cid_t in self._topology:
            target_mid = references[cid_t]
            self.clients[cid_s].add_to_aggregation_queue.remote(
                self.models[cid_t][target_mid]
            )

    def aggregate(self, client: Client) -> None:
        cid = ray.get(client.get_cid.remote())
        client_model = ray.get(client.get_model.remote())
        client_aggregation_queue = ray.get(client.get_aggregation_queue.remote())
        if len(client_aggregation_queue) == 0:
            logger.info("{} No models to aggregate".format(cid[:7]))
            return
        logger.info("{} Starting aggregation process".format(cid[:7]))
        logger.debug(
            "{} Aggregation queue: {}".format(
                cid[:7], [m.mid for m in client_aggregation_queue]
            )
        )
        # Aggregates the models in the aggregation queue
        # and sets the aggregated model as the current model
        aggregated_model = self._aggregation_function(
            [client_model] + client_aggregation_queue
        )
        logger.info("{} Aggregation finished".format(cid[:7]))
        ray.get(client.set_model.remote(aggregated_model))
        logger.info(
            "{cid} New model saved: {mid}".format(cid=cid[:7], mid=client_model.mid[:7])
        )

    def train(self) -> None:
        logger.info("Starting {} training process...".format(self._method_string))
        # Run the training process in parallel
        futures = [client.fit.remote() for client in self.clients.values()]
        ray.get(futures)
        logger.debug("{} training process finished".format(self._method_string))

    # Plotting and logging methods
    def plot_network(self, round) -> None:
        try:
            mpl_logger = logging.getLogger("matplotlib")
            mpl_logger.setLevel(logging.WARNING)
            plot_edges = [(i[:7], j[:7]) for i, j in self._topology]
            G = nx.DiGraph(plot_edges)
            plt.figure(figsize=(20, 12))  # Set the figure size to 8x8 inches
            options = {
                "font_size": 12,
                "node_size": 3250,
                "node_color": "white",
                "edgecolors": "black",
                "linewidths": 1,
                "width": 1,
            }
            nx.draw_networkx(G, **options)
            # Set margins for the axes so that nodes aren't clipped
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.title("Network topology: {}".format(round))
            # Log the network to tensorboard
            self._writer.add_figure("Network", plt.gcf(), global_step=round)
            logger.info("[TENSORBOARD] Added network plot to tensorboard")
        except Exception as e:
            logger.error("Error while plotting network: {}".format(e))
            logger.exception(e)
