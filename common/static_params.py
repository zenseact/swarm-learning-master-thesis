import torch
import torchvision
from enum import Enum
from datetime import datetime
from logging import INFO
from common.logger import log

class TASK(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class PartitionStrategy(Enum):
    RANDOM = 'random'
    LOCATION = 'location'

VM_IP = '172.25.16.67'
VM_KEY_PATH = '/home/nvidia/.ssh/id_rsa'

DEVICE_DICT = {
            #"agx4.nodes.edgelab.network" : 0, NOT WORKING ATM, fix it!! (flush and reinstall)
            "agx6.nodes.edgelab.network": 0,
            "agx9.nodes.edgelab.network": 0,
            "agx10.nodes.edgelab.network": 0,
            "orin1.nodes.edgelab.network": 0,
            "orin2.nodes.edgelab.network": 0
        }

TARGET_DISTANCES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
NUM_CLIENTS = 5
PERCENTAGE_OF_DATA = 0.02
NUM_OUTPUT = 51
IMG_SIZE = 224
RUN_PRETRAINED = True
BATCH_SIZE = 8
VAL_FACTOR = 0.1 # percentage of train data to use for validation
SUBSET_FACTOR = 0.003 # subset of test frames to use
USE_GPU = False
NUM_GLOBAL_ROUNDS = 3
NUM_LOCAL_EPOCHS = 4
PRINT_DEBUG_DATA = True
ML_TASK = TASK.REGRESSION
OUTPUT_SIZE = 66
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
STORED_GROUND_TRUTH_PATH = "/mnt/ZOD/ground_truth.json"
DATASET_ROOT = "/mnt/ZOD"
ZENSEACT_DATASET_ROOT = "/staging/dataset_donation/round_2"

available_device = 'cuda' if torch.cuda.is_available() else 'only cpu'
log(INFO,f"🚩{available_device} is available in this machine using PyTorch {torch.__version__}")
log(INFO,f"🚩Pytorch vision version: {torchvision.__version__}")


""" path to tensor board persistent folders"""
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
TB_PATH = f"runs{now}"
TB_CENTRALIZED_SUB_PATH = "Centralized/loss"
TB_FEDERATED_SUB_PATH = "Federated/loss"
TB_SWARM_SUB_PATH = "Swarm/loss"