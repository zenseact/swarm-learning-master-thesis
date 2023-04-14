from server_code.shared_dict import *

class EdgeHandler():
    def __init__(self, node_capacity: int, shared_dict_remote: SharedDict):
        self.node_capacity : int = node_capacity
        self.shared_dict_remote = shared_dict_remote

    def get_available_node(self):
        node = ray.get(self.shared_dict_remote.get_available_node.remote(self.node_capacity))
        return node
    
    def job_done(self, node : str):
        ray.get(self.shared_dict_remote.decrement.remote(node))
