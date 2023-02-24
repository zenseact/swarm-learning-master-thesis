from __future__ import annotations
from collections import OrderedDict
import importlib

import ray 
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import logging
import torch

from uuid import uuid4
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple
from torch.nn import Module
from copy import deepcopy
from torch import Tensor
from pathlib import Path
from ray.actor import ActorHandle

from utils import load_model
from utils.training import train, test
from utils.data.data_handler import DataObject

logger = logging.getLogger(__name__)

def run_swarm(
    config: dict,
    data: DataObject,
    log_dir: str,
) -> None:
    try:
        logger.info("Initialising swarm learning runtime")
        sim = SimulationRunner(
            config=config, 
            log_dir=log_dir,
            data=data,
            topology_generator=fully_connected_centralised, 
            aggregation_function=average,
        )
        logger.info("Swarm learning runtime initialised")
    except Exception as e:
        logger.error("Error initialising swarm learning runtime: {}".format(e))
        logger.exception(e)
        raise e
    
    sim.spawn_clients()
    sim.start()

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
        loss_function,
        torch_model: Module,
        log_dir: str,
    ) -> None:
        self.cid = cid
        self.icid = icid
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.model = TemporalModel(id, uuid4().hex, torch_model)
        self.loss_function = loss_function
        self.models = models
        self.aggregation_queue = []
        self.log_dir = log_dir
        self.round = 0
        
        if torch.cuda.is_available:
            logger.debug("Moving model to CUDA device")
            self.model.model = self.model.model.cuda()
        
        # Logging
        logging.basicConfig(
            filename="{}_{}.log".format(Path(self.log_dir, "swarm"), self.cid),
            encoding="utf-8",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    
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
        logger.debug("{cid} Adding model {mid} to history".format(
            cid=self.cid[:7],
            mid=self.model.mid[:7]))
        self.model.model.cpu()
        # Adds the current temporal model to the history
        self.models.update({self.model.mid: self.model})
        return self.cid, self.model.mid, self.models
    
    def add_to_aggregation_queue(self, model: TemporalModel) -> None:
        self.aggregation_queue.append(model)
        logger.debug("{cid} Added model {mid} to aggregation queue".format(
            cid=self.cid[:7],
            mid=model.mid[:7])
        )
    
    def set_model(self, model_weights: OrderedDict) -> None:
        model = deepcopy(self.model.model)
        model.load_state_dict(model_weights, strict=True)
        model.cuda()
        new_model_mid = uuid4().hex
        logger.debug("{cid} Setting new model {mid}".format(
            cid=self.cid[:7],
            mid=new_model_mid[:7]))
        self.model = TemporalModel(self.cid, new_model_mid, model)
        
    def fit(self) -> None:
        if self.round is None:
            print("IT IS NONE")
        self.round += 1
        logger.info("{} Starting training process on model {}".format(
            self.cid[:7],
            self.model.mid[:7]))
        mean_epoch_loss, batch_losses, mean_validation_loss, network = train(
            network=self.model.model,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=self.epochs,
            loss_function=self.loss_function,
            writer=SummaryWriter(self.log_dir),
            writer_path="swarm/loss/clients/{}/".format(self.icid),
            server_round=self.round,
        )
        logger.info("{} Training finished".format(self.cid[:7]))
        

class SimulationRunner:
    def __init__(
        self, 
        config: dict, 
        data: DataObject,
        log_dir: str,
        topology_generator: callable, 
        aggregation_function: callable,
    ) -> None:
        
        self.config = config
        self.topology_round = 0
        
        # Init the simulator
        logger.debug("Creating simulator instance...")
        self.sim = Simulator.remote(
            config=config,
            log_dir=log_dir,
            train_loaders=data.train.dataloaders,
            val_loaders=data.val.dataloaders,
            test_loader=data.test.dataloader,
            topology_generator=topology_generator,
            aggregation_function=aggregation_function,)
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
            logger.exception(e)
            raise e
        
    def set_topology(self) -> None:
        # Increment the topology number
        logging.debug("Topology round: {}".format(self.topology_round))
        self.topology_round += 1
        
        # Concurrently set the topology
        try:
            logger.info("Setting topology in ray workers...")
            logger.info("More information in the ray logs [swarm.log]")
            ray.get(self.sim.set_topology.remote())
            logger.debug("Topology set")
        except Exception as e:
            logger.error("Error setting topology: {}".format(e))
            logger.exception(e)
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
            logger.exception(e)
            raise e
        
        try:
            self.sim.plot_network.remote(self.topology_round)
        except Exception as e:
            logger.error("Error plotting network: {}".format(e))
            logger.exception(e)
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
            logger.exception(e)
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
        
    
    # THINGS
    def start(self) -> None:
        logger.info("Starting simulation loop")
        for i in range(0, self.config["swarm"]["global"]["global_rounds"]):
            logger.debug("Starting new round: {}".format(i + 1))
            start_time = datetime.now()
            
            self.train()
            self.set_topology()
            self.propagate_topology()
            self.start_communication()
            
            # Calculate the training duration
            end_time = datetime.now()
            soft_duration = str(end_time - start_time).split(".")[0]
            logger.debug("Round finished after {}".format(soft_duration))
        
        
@ray.remote
class Simulator:
    def __init__(
        self, 
        config: dict, 
        train_loaders: List[DataObject],
        val_loaders: List[DataObject],
        test_loader: DataObject,
        log_dir: str,
        topology_generator: callable, 
        aggregation_function: callable
    ) -> None:
        # Shared variables
        self.clients: Dict[str, ActorHandle[Client]] = {}
        self.models: Dict[str, TemporalModel] = {}
        
        # Sim properties
        self._n_clients = config["swarm"]["global"]["n_clients"]
        self._topology: List[str] = []
        self._topology_generator = topology_generator
        self._aggregation_function = aggregation_function
        self._config = config
        self._log_dir = log_dir
        self._train_loaders = train_loaders
        self._val_loaders = val_loaders
        self._test_loader = test_loader
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self._client_resources = config["swarm"]["global"]["client_resources"]
        
        # Load the loss function
        loss_method = self._config["model"]["loss"]
        module = importlib.import_module("torch.nn")
        self._loss_function = getattr(module, loss_method)
        
        # Logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename="{}.log".format(Path(self._log_dir, "swarm")),
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
    def set_topology(self):
        self._topology = self._topology_generator(list(self.clients.keys()))
    
    # Support methods
    def create_client_actor(self, index: int) -> Tuple[str, Client]:
        id = uuid4().hex
        self.models[id] = {}
        model = load_model(self._config)
        
        logger.info("Created client with short id: {}".format(id[:7]))
        client_ref = Client.options(**self._client_resources).remote(
            cid = id,
            icid=index,
            trainloader=self._train_loaders[index],
            valloader=self._val_loaders[index],
            epochs=self._config["swarm"]["client"]["epochs"],
            models=self.models[id], 
            loss_function=self._loss_function,
            torch_model=model,
            log_dir=self._log_dir,
        )
        return id, client_ref
    
    def spawn_clients(self) -> None:
        try:
            logger.info("Spawning client actors...")
            self.clients.update([self.create_client_actor(i) for i in range(self._n_clients)])
            logger.info("Client actors spawned")
        except Exception as e:
            logger.error("Error while spawning clients: {}".format(e))
            logger.exception(e)
            raise e
        
    def send_intent(self) -> List[object]:
        try:
            logger.debug("Sending intents...")
            futures = [client_actor.push_to_history.remote() for client_actor in self.clients.values()]
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
            logger.exception(e)
            raise e
    
    def send_jobs(self, references: dict) -> None:
        logger.debug("Sending jobs...")
        for cid_s, cid_t in self._topology:
            target_mid = references[cid_t]
            self.clients[cid_s].add_to_aggregation_queue.remote(self.models[cid_t][target_mid])
            
    def aggregate(self, client: Client) -> None:
        cid = ray.get(client.get_cid.remote())
        client_model = ray.get(client.get_model.remote())
        client_aggregation_queue = ray.get(client.get_aggregation_queue.remote())
        logger.info("{} Starting aggregation process".format(cid[:7]))
        # Aggregates the models in the aggregation queue
        # and sets the aggregated model as the current model
        aggregated_model = self._aggregation_function([client_model] + client_aggregation_queue)
        logger.info("{} Aggregation finished".format(cid[:7]))
        ray.get(client.set_model.remote(aggregated_model))
        logger.info("{cid} New model saved: {mid}".format(
            cid=cid[:7],
            mid=client_model.mid[:7]))
    
    def train(self) -> None:
        logger.info("Starting swarm training process...")
        # Run the training process in parallel
        futures = [client.fit.remote() for client in self.clients.values()]
        ray.get(futures)
        logger.debug("Swarm training process finished")

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
            self._writer.add_figure("Network", plt.gcf())
            logger.info("[TENSORBOARD] Added network plot to tensorboard")
        except Exception as e:
            logger.error("Error while plotting network: {}".format(e))
            logger.exception(e)
        

def fully_connected_centralised(clients: List[str]) -> None:
    return list(itertools.permutations(clients, 2))

def average(models: List[TemporalModel] = None) -> OrderedDict:
    # Average factor k, and a deepcopy of the first model
    k = 1/len(models)
    new_model = deepcopy(models[0].get_parameters())
 
    # Sum the model weights
    for target_model in models:
        target_parameters = target_model.get_parameters()
        for key in new_model.keys():
            new_model[key] += target_parameters[key]

    # Multiply by k to get the average
    for key in new_model.keys():
        new_model[key] *= k
       
    return new_model
