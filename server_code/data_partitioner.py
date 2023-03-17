from common.static_params import PartitionStrategy
from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
import random
import numpy as np

# Data partition will be saved as a dictionary client_cid -> [frames_id's] and this dict is 
# downloaded by the client that loads the correct elements by the idx list in the dictionary

# random.seed(2023)

#load data based on cid and strategy
def partition_train_data(strat: PartitionStrategy, no_clients: int):
    version = "full"  # "mini" or "full"
    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version=version)
    training_frames_all = zod_frames.get_split(constants.TRAIN)
    validation_frames_all = zod_frames.get_split(constants.VAL)
    
    #put train and val together, clients will partition themselves.
    training_frames_all = training_frames_all+validation_frames_all

    if strat == PartitionStrategy.RANDOM:
        # cids will be 0,1..N
        cid_partitions = {}
        random.shuffle(training_frames_all)
        sublist_size = len(training_frames_all) // no_clients
        for i in range(no_clients):
            cid_partitions[i] = training_frames_all[i*sublist_size:(i+1)*sublist_size]
        np.savez('../tmp/partitions.npz', **cid_partitions)
    
    if strat == PartitionStrategy.LOCATION:
        pass