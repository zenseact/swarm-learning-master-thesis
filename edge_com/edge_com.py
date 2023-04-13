import os
import numpy as np
from edge_com.edge_handler import EdgeHandler
import time
from edge_com.client_run import run

class EdgeCom:
    def __init__(self, edge_handler: EdgeHandler):
        self.edge_handler: EdgeHandler = edge_handler
        self.node : str = None
    
    def update_model(self, cid: str):
        # find available edge node and tell it to train
        self.__train(cid)
        # wait for edge node to finish and fetch new model
        model = self.__recieve(cid)
        # remove the model after fetch
        self.__remove(cid)
        return model

    def __train(self, cid: str):
        print('telling client to train with parameter cid')
        self.node = self.edge_handler.get_available_node()
        run(self.node, cid)
            

    def __recieve(self, cid: str):
        print('Waiting to recieve file from client')
        file_recieved = False
        while not file_recieved:
            file_recieved = os.path.isfile("tmp/res"+cid+".npz")
        model = np.load("tmp/res"+cid+".npz")
        print('file recieved from client')
        return model

    def __remove(self, cid: str):
        self.edge_handler.job_done(self.node)
        self.node = None
        os.remove("tmp/res"+cid+".npz")
        print('client file removed')
