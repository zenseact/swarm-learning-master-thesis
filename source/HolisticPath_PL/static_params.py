from imports import *

# read unused frames
with open("frames_with_less_than_165m_hp.json") as f: UNUSED_FRAMES = set(json.load(f))

# read the config
with open("config.json") as f: configs = json.load(f)
print(sys.argv)
if(len(sys.argv) <= 1 or len(sys.argv[1]) > 2): 
    config = configs[-1]
else:
    config = [c for c in configs if c['exp_id'] == int(sys.argv[1])][0]
print(config)

# helper function to read from config
c = lambda a : config[a]

# specify the device
DEVICE = torch.device("cuda" if c('use_gpu') else "cpu")

print(f"PyTorch={torch.__version__}. Pytorch vision={torchvision.__version__}. Flower={fl.__version__}")
print(f"Training will run on: {DEVICE}s")

# path to tensor board persistent folders
DISC = f"exp-{c('exp_id')}_{c('type')}_agent-{c('agent_id')}_{c('model')}_{c('dataset_division')}_{c('loss')}_{c('target_distances')[-1]}m_imgnet_normalized_{c('num_local_epochs')}epochs_lr{c('learning_rate')}_{c('subset_factor')*34000}trainImages_bs{c('batch_size')}_imgSize{c('image_size')}_unfreezed_ema-{c('use_ema')}"
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
TB_PATH = f"TensorBoard/{DISC}_{now}"
TB_CENTRALIZED_SUB_PATH = "TensorBoard_Centralized/loss/"
TB_FEDERATED_SUB_PATH = "TensorBoard_Federated/loss/"
TB_SWARM_SUB_PATH = "TensorBoard_Swarm/loss/"
print(DISC)

# global tensorboard writer
writer = SummaryWriter(TB_PATH)
