import ctypes
import multiprocessing as mp
import numpy as np

def create_shared_dict():
    # Define the dictionary
    my_dict = {'key1': 'value1', 'key2': 'value2'}
    
    # Create a shared memory buffer to hold the dictionary
    buffer = mp.RawArray(ctypes.c_char, len(repr(my_dict).encode()))
    
    # Copy the dictionary into the shared memory buffer
    ctypes.memmove(buffer, repr(my_dict).encode(), len(repr(my_dict).encode()))
    
    return buffer

# Use the shared dictionary from another process
def use_shared_dict(buffer):
    # Access the shared memory buffer as a numpy array
    arr = np.frombuffer(buffer, dtype=np.uint8)
    shared_dict = arr.view(dtype=np.dtype('O')).reshape(1)[0]
    
    # Convert the shared dictionary back into a regular dictionary
    my_dict = eval(shared_dict.tobytes().decode())
    
    # Access and modify the dictionary as needed
    return my_dict