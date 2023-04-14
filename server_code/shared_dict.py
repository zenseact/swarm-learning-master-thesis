import ray
import time

# Create a shared dictionary
@ray.remote
class SharedDict:
    def __init__(self, d):
        self.dict = d
        self.lock = ray.Lock()
    
    def get(self):
        return self.dict
    
    def set(self, key, value):
        self.dict[key] = value
    
    def decrement(self, key):
        with self.lock:
            self.dict[key] = self.dict[key]-1

    def get_available_node(self, node_capacity):
        with self.lock:
            node, running = min(self.dict.items(), key=lambda x: x[1])
            if running < node_capacity:
                self.dict[node] = self.dict[node]+1
                return node
            else:
                time.sleep(10)
                return self.get_available_node(node_capacity)