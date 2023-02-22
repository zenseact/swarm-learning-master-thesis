from imports import *


class TASK(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


NUM_CLIENTS = 2
NUM_OUTPUT = 66
IMG_SIZE = 160
RUN_PRETRAINED = True
BATCH_SIZE = 8
VAL_FACTOR = 0.1
SUBSET_FACTOR = 0.003
USE_GPU = False
NUM_GLOBAL_ROUNDS = 3
NUM_LOCAL_EPOCHS = 10
PRINT_DEBUG_DATA = True
ML_TASK = TASK.REGRESSION
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

available_device = 'cuda' if torch.cuda.is_available() else 'only cpu'
print(f"🚩{available_device} is available in this machine using PyTorch {torch.__version__} and Flower {fl.__version__}")
print(f"🚩Pytorch vision version: {torchvision.__version__}")
print(f'🚩training will run on: {DEVICE}s')
