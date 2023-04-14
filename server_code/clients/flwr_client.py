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
        print(f"⤺ Get model parameters of client {self.cid}]")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print('getting parameters from edge device')
        params = self.edge_com.update_model(self.cid)
        # return 1 as nr of training examples for now
        return params, 1, {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        if (not self.net):
            self.net = load_model(self.cid)
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        if (ML_TASK == TASK.CLASSIFICATION):
            print(f"🌠 [Client {self.cid}] test loss {loss}, accuracy {accuracy}")
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        else:
            print(f"🌠 [Client {self.cid}] test RMSE {loss}")
            return float(loss), len(self.valloader), {"loss": float(loss)}