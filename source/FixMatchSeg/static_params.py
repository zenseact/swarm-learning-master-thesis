from imports import *

# read the config
with open("config.json") as f: configs = json.load(f)
print(sys.argv)
if(len(sys.argv) <= 1 or len(sys.argv[1]) > 2): 
    config = configs[-1]
else:
    config = [c for c in configs if c['exp_id'] == int(sys.argv[1])][0]
print(config)

# helper function to read from config
c = lambda a: config.get(a, None) 

IMG_SIZE = c('image_size')
RUN_PRETRAINED = c('run_pretrained')
BATCH_SIZE = c('batch_size')
VAL_FACTOR = c('val_factor')
SUBSET_FACTOR = c('subset_factor')
LABELED_FACTOR = c('labeled_factor')
USE_GPU = c('use_gpu')
NUM_CLIENTS = c('num_clients')
NUM_GLOBAL_ROUNDS = c('num_global_rounds')
NUM_LOCAL_EPOCHS = c('num_local_epochs')
LR=c('learning_rate')
NUM_WORKERS = c('num_workers')
PREFETCH_FACTOR = c('prefetch_factor')
FRAMES_IMAGE_MEAN = c('frames_image_mean')
FRAMES_IMAGE_STD = c('frames_image_std')
IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]
SEED=42
GPU_ID = c('gpu_id')
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
TRAIN_FRAMES_PATH = "GroundTruth/training_seg_annotated_frames.json"
VAL_FRAMES_PATH = "GroundTruth/validation_seg_annotated_frames.json"
DATASET_ROOT = "/mnt/ZOD"
ZENSEACT_DATASET_ROOT = "/staging/dataset_donation/round_2"

# specify the device
DEVICE = torch.device("cuda" if c('use_gpu') else "cpu")

print(f"PyTorch={torch.__version__}. Pytorch vision={torchvision.__version__}. Flower={fl.__version__}")
print(f"Training will run on: {DEVICE}s")

# path to tensor board persistent folders
DISC = f"exp-{c('exp_id')}_{c('type')}_imgnet_normalized_{c('num_local_epochs')}epochs_lr{c('learning_rate')}_{c('subset_factor')*52198}trainImages_bs{c('batch_size')}_imgSize{c('image_size')}"
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
TB_PATH = f"TensorBoard/{now}_{DISC}/"
TB_CENTRALIZED_SUB_PATH = "TensorBoard_Centralized/loss/"
TB_FEDERATED_SUB_PATH = "TensorBoard_Federated/loss/"
TB_SWARM_SUB_PATH = "TensorBoard_Swarm/loss/"
print(f'{DISC}_{now}')

# global tensorboard writer
writer = SummaryWriter(TB_PATH)
