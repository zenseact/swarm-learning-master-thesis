import time
from server_code.shared_dict import use_shared_dict
import random

class EdgeHandler():
    def __init__(self, node_capacity: int, buffer):
        self.node_capacity : int = node_capacity
        self.buffer = buffer

    # hack solution for synchronization in here..
    def get_available_node(self):
        time.sleep(random.uniform(0, 10))
        nodes_running = use_shared_dict(self.buffer)
        node, running = min(nodes_running.items(), key=lambda x: x[1])
        if running < self.node_capacity:
            nodes_running[node] = nodes_running[node] + 1
            time.sleep(2)
            if nodes_running[node] <= self.node_capacity:
                return node
            else:
                nodes_running[node] = nodes_running[node] - 1
                return self.get_available_node()
        return self.get_available_node()
    
    def job_done(self, node : str):
        nodes_running = use_shared_dict(self.buffer)
        nodes_running[node] = nodes_running[node] - 1
