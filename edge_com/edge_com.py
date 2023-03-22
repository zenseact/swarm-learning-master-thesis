import os
import numpy as np
from edge_com.edge_handler import EdgeHandler

class EdgeCom:
    def __init__(self, edge_handler: EdgeHandler):
        self.edge_handler: EdgeHandler = edge_handler
        self.node : str = None
    
    def update_model(self, cid: str, strategy: str):
        # find available edge node and tell it to train
        self.__ping_train(cid, strategy)
        # wait for edge node to finish and fetch new model
        model = self.__recieve(cid)
        # remove the model after fetch
        self.__remove(cid)
        return model

    def __ping_train(self, cid: str, strategy: str):
        self.node = self.edge_handler.get_available_node()
        response = -1
        while response != 0:
            response = os.system(f"ping -c 1 -M ip --ip-data={cid}:{strategy} {self.node}")

    def __recieve(self, cid: str):
        file_recieved = False
        while not file_recieved:
            file_recieved = os.path.isfile("res.npz")
        model = np.load("../tmp/res"+cid+".npz")
        return model

    def __remove(self, cid: str):
        self.edge_handler.job_done(self.node)
        self.node = None
        os.remove("../tmp/res"+cid+".npz")
        print('done')
