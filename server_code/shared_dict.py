import multiprocessing as mp
import numpy as np


# Create the shared dictionary in a separate process
def create_shared_dict():
    # Define the dictionary
    my_dict = {
            #"agx4.nodes.edgelab.network" : 0, NOT WORKING ATM, fix it!! (flush and reinstall)
            "agx6.nodes.edgelab.network" : 0,
            "agx9.nodes.edgelab.network" : 0,
            "agx10.nodes.edgelab.network" : 0,
            "orin1.nodes.edgelab.network" : 0,
            "orin2.nodes.edgelab.network" : 0
        }
    
    # Create a shared memory buffer to hold the dictionary
    dt = np.dtype([('key', 'S'), ('value', np.int32)])
    buffer = mp.RawArray('c', len(my_dict) * dt.itemsize)
    arr = np.frombuffer(buffer, dtype=dt)
    shared_dict = {arr[i]['key'].decode(): arr[i]['value'] for i in range(len(my_dict))}
    
    # Copy the dictionary into the shared memory buffer
    shared_dict[:] = repr(my_dict).encode()
    
    return buffer

# Use the shared dictionary from another process
def use_shared_dict(buffer):
    dt = np.dtype([('key', 'S'), ('value', np.int32)])
    # Access the shared memory buffer as a numpy array
    arr = np.frombuffer(buffer, dtype=dt)
    shared_dict = arr.view(dtype=np.dtype('O')).reshape(1)[0]
    
    # Convert the shared dictionary back into a regular dictionary
    my_dict = eval(shared_dict.tobytes().decode())
    
    return my_dict