from imports import *

NUM_OUTPUT = 51
IMG_SIZE = 500
RUN_PRETRAINED = True
BATCH_SIZE = 32
VAL_FACTOR = 0.15
SUBSET_FACTOR = 1
USE_GPU = True
NUM_GLOBAL_ROUNDS = 3
NUM_LOCAL_EPOCHS = 10
PRINT_DEBUG_DATA = True
NUM_WORKERS = os.cpu_count()
FRAMES_IMAGE_MEAN = [0.337, 0.345, 0.367]
FRAMES_IMAGE_STD = [0.160, 0.180, 0.214]
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
TRAIN_FRAMES_PATH = "GroundTruth/training_seg_annotated_frames.json"
VAL_FRAMES_PATH = "GroundTruth/validation_seg_annotated_frames.json"
DATASET_ROOT = "/mnt/ZOD"
ZENSEACT_DATASET_ROOT = "/staging/dataset_donation/round_2"

available_device = "cuda" if torch.cuda.is_available() else "only cpu"
print(f"🚩{available_device} is available in this machine using PyTorch {torch.__version__} and Flower {fl.__version__}")
print(f"🚩Pytorch vision version: {torchvision.__version__}")
print(f"🚩training will run on: {DEVICE}s")

""" path to tensor board persistent folders"""
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
TB_PATH = f"TensorBoard/runs{now}"
TB_CENTRALIZED_SUB_PATH = "TensorBoard_Centralized/loss"
TB_FEDERATED_SUB_PATH = "TensorBoard_Federated/loss"
TB_SWARM_SUB_PATH = "TensorBoard_Swarm/loss"