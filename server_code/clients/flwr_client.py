from common.utilities import *
from common.static_params import *
from edge_com.edge_handler import EdgeHandler
from edge_com.edge_com import EdgeCom
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, edge_handler: EdgeHandler, tb_path=None, federated_subpath=None):
        self.cid = cid
        self.edge_com = EdgeCom(edge_handler)
        self.losses = []
        self.val_losses = []
        self.net = None
        #self.tb_path = tb_path
        #self.federated_subpath = federated_subpath
        #self.tb_writer = SummaryWriter(self.tb_path)s

    def get_parameters(self, config):
        log(INFO,f"⤺ Get model parameters of client {self.cid}]")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        log(INFO,'Starting edge devie training process for cid: {self.cid}')
        params = self.edge_com.update_model(self.cid)
        # return 1 as nr of training examples for now
        return params, 1, {}