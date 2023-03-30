from common.static_params import PartitionStrategy
from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
import random
import numpy as np
from common.static_params import STORED_GROUND_TRUTH_PATH
from common.groundtruth_utils import load_ground_truth, get_ground_truth, OUTPUT_SIZE

# Data partition will be saved as a dictionary client_cid -> [frames_id's] and this dict is 
# downloaded by the client that loads the correct elements by the idx list in the dictionary

# random.seed(2023)

#load data based on cid and strategy
def partition_train_data(strat: PartitionStrategy, no_clients: int):
    version = "full"  # "mini" or "full"
    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version=version)
    training_frames_all = zod_frames.get_split(constants.TRAIN)
    validation_frames_all = zod_frames.get_split(constants.VAL)
    
    ground_truth = load_ground_truth(STORED_GROUND_TRUTH_PATH)
    print('loaded stored ground truth')

    training_frames_all = [idx for idx in training_frames_all if is_valid_frame(idx, ground_truth)]
    validation_frames_all = [idx for idx in validation_frames_all if is_valid_frame(idx, ground_truth)]

    #put train and val together, clients will partition themselves.
    training_frames_all = training_frames_all+validation_frames_all

    if strat == PartitionStrategy.RANDOM:
        # cids are 0,1,2..N
        cid_partitions = {}
        random.shuffle(training_frames_all)
        sublist_size = len(training_frames_all) // no_clients
        for i in range(no_clients):
            cid_partitions[str(i)] = training_frames_all[i*sublist_size:(i+1)*sublist_size]
        # save partitions for clients to download
        np.savez('tmp/partitions.npz', **cid_partitions)
        return cid_partitions
    if strat == PartitionStrategy.LOCATION:
        pass

def is_valid_frame(frame_id, ground_truth):
    return frame_id in ground_truth