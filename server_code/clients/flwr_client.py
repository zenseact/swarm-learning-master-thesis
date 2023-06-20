from edge_com.edge_handler import EdgeHandler
from edge_com.edge_com import EdgeCom
import flwr as fl
from common.logger import fleet_log
from logging import INFO

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

    def fit(self, parameters, config):
        fleet_log(INFO,f'Starting edge devie training process for cid: {self.cid}')
        params = self.edge_com.update_model(self.cid)
        # return 1 as nr of training examples for now
        return params, 1, {}