from fleetlearning.common.static_params import PartitionStrategy
from zod import ZodFrames
import zod.constants as constants
import random
import numpy as np
from fleetlearning.common.static_params import global_configs
from fleetlearning.common.groundtruth_utils import load_ground_truth
from fleetlearning.common.logger import fleet_log
from logging import INFO
from fleetlearning.common.datasets import ZODImporter

# Data partition will be saved as a dictionary client_cid -> [frames_id's] and this dict is
# downloaded by the client that loads the correct elements by the idx list in the dictionary

random.seed(2023)


# load data based on cid and strategy
def partition_train_data(
    strat: PartitionStrategy, no_clients: int, zod_importer: ZODImporter, ip_maps: dict
):
    zod_frames = zod_importer.zod_frames
    training_frames_all = zod_frames.get_split(constants.TRAIN)

    ground_truth = load_ground_truth(global_configs.STORED_GROUND_TRUTH_PATH)
    fleet_log(INFO, "loaded stored ground truth")

    training_frames_all = [
        idx for idx in training_frames_all if is_valid_frame(idx, ground_truth)
    ]

    # put train and val together, clients will partition themselves.
    training_frames_all

    if strat == PartitionStrategy.RANDOM:
        cid_partitions = {}
        random.shuffle(training_frames_all)
        sublist_size = len(training_frames_all) // no_clients
        for i, client_ID in enumerate(ip_maps["client_id_to_node_map"].keys()):
            cid_partitions[str(client_ID)] = training_frames_all[
                i * sublist_size : (i + 1) * sublist_size
            ]
        # save partitions for clients to download
        np.savez("tmp/partitions.npz", **cid_partitions)
    if strat == PartitionStrategy.LOCATION:
        pass

    return cid_partitions


def is_valid_frame(frame_id, ground_truth):
    return frame_id in ground_truth
