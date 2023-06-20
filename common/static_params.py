import torch
import torchvision
from enum import Enum
from datetime import datetime
from logging import INFO
from common.logger import fleet_log
from dataclasses import dataclass

class TASK(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class PartitionStrategy(Enum):
    RANDOM = 'random'
    LOCATION = 'location'

@dataclass
class GlobalConfigs:
    def __init__(self):
        self.SERVER_MAIN_PATH = "/root/Fleet/fleet-learning/"
        self.VM_IP = '172.25.16.67'
        self.VM_KEY_PATH = '/home/nvidia/.ssh/id_rsa'

        self.DEVICE_DICT = {
                    # "agx4.nodes.edgelab.network" : 0, NOT WORKING ATM, fix it!! (flush and reinstall)
                    "agx6.nodes.edgelab.network": 0,
                    "agx9.nodes.edgelab.network": 0,
                    "agx10.nodes.edgelab.network": 0,
                    "orin1.nodes.edgelab.network": 0,
                    "orin2.nodes.edgelab.network": 0
                }

        self.TARGET_DISTANCES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
        self.NUM_CLIENTS = 5
        self.PERCENTAGE_OF_DATA = 0.02
        self.NUM_OUTPUT = 51
        self.IMG_SIZE = 256
        self.RUN_PRETRAINED = False
        self.BATCH_SIZE = 8
        self.VAL_FACTOR = 0.1 # percentage of train data to use for validation
        self.SUBSET_FACTOR = 0.003 # subset of test frames to use
        self.USE_GPU = False
        self.NUM_GLOBAL_ROUNDS = 3
        self.NUM_LOCAL_EPOCHS = 4
        self.PRINT_DEBUG_DATA = True
        self.ML_TASK = TASK.REGRESSION
        self.OUTPUT_SIZE = 66
        self.DEVICE = torch.device('cuda' if self.USE_GPU else 'cpu')
        self.STORED_GROUND_TRUTH_PATH = "/mnt/ZOD/ground_truth.json"
        self.DATASET_ROOT = "/mnt/ZOD"
        self.ZENSEACT_DATASET_ROOT = "/staging/dataset_donation/round_2"

        self.available_device = 'cuda' if torch.cuda.is_available() else 'only cpu'
        fleet_log(INFO,f"🚩{self.available_device} is available in this machine using PyTorch {torch.__version__}")
        fleet_log(INFO,f"🚩Pytorch vision version: {torchvision.__version__}")


        """ path to tensor board persistent folders"""
        self.now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.TB_PATH = f"runs{self.now}"
        self.TB_CENTRALIZED_SUB_PATH = "Centralized/loss"
        self.TB_FEDERATED_SUB_PATH = "Federated/loss"
        self.TB_SWARM_SUB_PATH = "Swarm/loss"

global_configs = GlobalConfigs()