import time

class EdgeHandler():
    def __init__(self, node_capacity: int):
        self.node_capacity : int = node_capacity
        self.nodes_running : dict = {
            #"agx4.nodes.edgelab.network" : 0, NOT WORKING ATM, fix it!! (flush and reinstall)
            "agx6.nodes.edgelab.network" : 0,
            "agx9.nodes.edgelab.network" : 0,
            "agx10.nodes.edgelab.network" : 0,
            "orin1.nodes.edgelab.network" : 0,
            "orin2.nodes.edgelab.network" : 0
        }

    
    def get_available_node(self):    
        node, running = min(self.nodes_running.items(), key=lambda x: x[1])
        if running < self.node_capacity:
            self.nodes_running[node] = self.nodes_running[node] + 1
            # synchronization tricky in ray so we do hack lock instead..
            time.sleep(2)
            if self.nodes_running[node] <= self.node_capacity:
                return node
            else:
                self.nodes_running[node] = self.nodes_running[node] - 1
                return self.get_available_node()
        return self.get_available_node()
    
    def job_done(self, node : str):
        self.nodes_running[node] = self.nodes_running[node] - 1
