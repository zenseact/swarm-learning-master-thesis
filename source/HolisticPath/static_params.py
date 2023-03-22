from imports import *


class TASK(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


NUM_CLIENTS = 2
NUM_OUTPUT = 66
IMG_SIZE = 700
RUN_PRETRAINED = True
BATCH_SIZE = 64
VAL_FACTOR = 0.1
SUBSET_FACTOR = 0.003
USE_GPU = True
NUM_GLOBAL_ROUNDS = 3
NUM_LOCAL_EPOCHS = 10
PRINT_DEBUG_DATA = True
NUM_WORKERS=0
ML_TASK = TASK.REGRESSION
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
STORED_GROUND_TRUTH_PATH = "cached_gt/hp_gt.json"
STORED_BALANCED_DS_PATH = "cached_gt/balanced_frames.txt"
DATASET_ROOT = "/mnt/ZOD"
ZENSEACT_DATASET_ROOT = "/staging/dataset_donation/round_2"

available_device = 'cuda' if torch.cuda.is_available() else 'only cpu'
print(f"🚩{available_device} is available in this machine using PyTorch {torch.__version__} and Flower {fl.__version__}")
print(f"🚩Pytorch vision version: {torchvision.__version__}")
print(f'🚩training will run on: {DEVICE}s')


""" path to tensor board persistent folders"""
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
TB_PATH = f"runs{now}"
TB_CENTRALIZED_SUB_PATH = "Centralized/loss"
TB_FEDERATED_SUB_PATH = "Federated/loss"
TB_SWARM_SUB_PATH = "Swarm/loss"


CORRUPTED_FRAMES=['094045', '086941', '095594', '092027', '070585', '087311', '085515', '095438', '087558', '053561', '099029', '095461', '011701', '006086', '063752', '097248', '028126', '091005', '038095', '081670', '053583', '085665', '089688', '005350', '065644', '011950', '098939', '019180', '081417', '006820', '024706', '096708', '015361', '058768', '061939', '055269', '027010', '085884', '013722', '053300', '084894', '056046', '042235', '088568', '030137', '062424', '079558', '058543', '019863', '005063', '052067', '058355', '073891', '010961', '081841', '059038', '075155', '056047', '055799', '060196', '026807', '083883', '093495', '033850', '055871', '011671', '068151', '059138', '096970', '057623', '095901', '094430', '060380', '060170', '053988', '042345', '066143', '088890', '056350', '056602', '081452', '060371', '059245', '089308', '056802', '056089', '074612', '054693', '094180', '041921', '037456', '008938', '059914', '055775', '053832', '058200', '057122', '020843', '082524', '057752', '098660', '049220', '053839', '055192', '074967', '061009', '086327', '003128', '095253', '055872', '061742', '053712', '007109', '089292', '075997', '016241', '062825', '061408', '056191', '076380', '054592', '051924', '055306', '019397', '078544', '097724', '085054', '083647']