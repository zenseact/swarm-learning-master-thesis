import os
import numpy as np
from edge_com.edge_handler import EdgeHandler
import time

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
        print('pinging client waiting for response')
        self.node = self.edge_handler.get_available_node()
        response = -1
        while response != 0:
            ip_data = f"{cid}:{strategy}"
            hex_data = ''.join(hex(ord(c))[2:] for c in ip_data)
            response = os.system(f"ping -c 1 -M dont -p {hex_data} {self.node}")
            time.sleep(1)
        print('ping recieved by client')
            

    def __recieve(self, cid: str):
        print('Waiting to recieve file from client')
        file_recieved = False
        while not file_recieved:
            file_recieved = os.path.isfile("res"+cid+".npz")
        model = np.load("../tmp/res"+cid+".npz")
        print('file recieved from client')
        return model

    def __remove(self, cid: str):
        self.edge_handler.job_done(self.node)
        self.node = None
        os.remove("../tmp/res"+cid+".npz")
        print('client file removed')
