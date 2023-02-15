from __future__ import annotations

import importlib
import itertools
import logging
import numpy as np
import ray

from collections import OrderedDict
from datetime import datetime
from torch.nn import Module
from torch import Tensor, Size
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from uuid import uuid4

from utils import load_model, ZODHandler as Data
from utils.training import train, test


def run_federated(
    config: dict,
    training_loaders: List[DataLoader],
    validation_loaders: List[DataLoader],
    test_loader: DataLoader,
    log_dir: str,
) -> None:
    pass


class TemporalModel:
    # A class which saves the parameters of a model and the time it was created
    # Every temporal model has a unique id (mid)
    def __init__(self, mid, parameters) -> None:
        self.mid = mid
        self.parameters = parameters
        self.created_at = datetime.now()

class SwarmSimulator:
    def __init__(self, config: dict, data: Data = None) -> None:
        self._clients = dict()
        self.data = data
        self.spwan_clients(config["swarm"]["global"]["n_clients"])
    
    def spwan_clients(self, n_clients: int) -> None:
        for _ in range(n_clients):
            cid = uuid4().hex
            self._clients.update({cid: SwarmClient.remote(cid, self)})
            
    def set_topology(self, topology: List[Tuple]) -> None:
        self.topology = topology
        
    def start_simulation(self) -> None:
        assert self.topology is not None, "Topology not set"
        while len(self.topology) > 0:
            pass
    
    @property
    def clients(self) -> List:
        return self._clients
        
@ray.remote
class SwarmClient:
    def __init__(
        self,
        cid: int, 
        initial_weights: List,
        save_all: bool = False
    ) -> None:
        self.cid = cid
        self._model = TemporalModel(uuid4().hex, initial_weights)
        self.models = {self._model.mid: self._model}
        self._model_history = dict()
        self._save_all = save_all
        
        self.push_to_history()

    def ping(self, target: SwarmClient) -> str:
        # Saves the current temporal model of the target client to their history
        # and returns the id of the current temporal model of the target client
        return ray.get(target.push_to_history.remote())
    
    def fetch(self, target: SwarmClient, mid: str) -> TemporalModel:
        # Returns the temporal model of the target client with the given id
        return ray.get(target.find_model.remote(mid))
    
    def push_to_history(self) -> str:
        # Adds the current temporal model to the history
        self.models.update({self.model.mid: self.model})
        return self.model.mid

    def find_model(self, mid: str) -> List:
        # Returns the temporal model with the given id
        return self.models.get(mid)

    def get_model(self) -> TemporalModel:
        # Calls the getter of the model property
        return self.model
    
    def get_models(self) -> List:
        # Returns the models stored by the client
        return self.models
    
    def set_model(self, parameters) -> None:
        # Calls the setter of the model property
        self.model = parameters
        
    def get_model_history(self) -> List:
        # Calls the getter of the model_history property
        return self.model_history
    
    def train(self) -> Tuple:
        epoch_loss, batch_losses = train(
            network=self.model,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=None, # TODO: Read from config
            loss_function=None, # TODO: Read from config
            writer=None, # TODO: Read from config
            writer_path=None, # TODO: Read from config
            server_round=None, # TODO: Read from config
        )
    
    @property
    def model(self) -> TemporalModel:
        # Returns the current temporal model. If save_all is True, returns the last temporal model
        if self._save_all:
            return self.model_history[-1]
        else:
            return self._model
    
    @model.setter
    def model(self, parameters) -> None:
        # Creates a new temporal model with the given parameters
        # if save_all is True, saves the new temporal model to the history
        mid = uuid4().hex
        if self._save_all:
            self.models.update({mid: TemporalModel(mid, parameters)})
        else:
            self._model = TemporalModel(mid, parameters)
    
    @property
    def model_history(self) -> List:
        # Returns the history of temporal models sorted by creation time
        return sorted(list(self.models.values()), key=lambda x: self.models[x.mid].created_at)


def simulate_training(weights):
    weights

def fully_connected_centralised(clients: List) -> None:
    return list(itertools.product(clients, clients))
