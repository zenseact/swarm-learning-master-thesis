from static_params import *
from fixmatch_utils import *

class ZODImporter:
    def __init__(
        self,
        root=DATASET_ROOT,
        subset_factor=SUBSET_FACTOR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        tb_path=TB_PATH,
        zod_frames=None,
        training_frames=None, 
        validation_frames=None
    ):
        if(zod_frames == None):
            self.zod_frames, self.training_frames_all, self.validation_frames_all = load_ego_road(root)

            self.training_frames = self.training_frames_all[: int(len(self.training_frames_all) * subset_factor)]
            self.validation_frames = self.validation_frames_all[: int(len(self.validation_frames_all) * subset_factor)]

            self.training_frames = [x for x in tqdm(self.training_frames) if self.is_valid(x)]
            self.validation_frames = [x for x in tqdm(self.validation_frames) if self.is_valid(x)]
        else:
            self.zod_frames = zod_frames
            self.training_frames = training_frames
            self.validation_frames = validation_frames
            
        print("length of training_frames subset:", len(self.training_frames))
        print("length of test_frames subset:", len(self.validation_frames))

        self.img_size = img_size
        self.batch_size = batch_size
        self.tb_path = tb_path

        
    def is_valid(self, frame_id):
        zod_frame = self.zod_frames[frame_id]
        
        try:
            # get the ego road annotations
            polygon_annotations = zod_frame.get_annotation(AnnotationProject.EGO_ROAD)

            # convert the polygons to a binary mask (which can be used
            mask = polygons_to_binary_mask(polygon_annotations)
        except:
            print(f'{frame_id} is invalid')
            return False
        return True
        
    def get_zod_datasets(self):
        transform_labeled = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        transform_fixmatch = TransformFixMatch(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        train_ids, validation_ids = x_u_split(self.training_frames, self.validation_frames)

        train_dataset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=train_ids, set_type='train', transform=transform_labeled, transform_fixmatch=transform_fixmatch)
        validation_set = ZodDataset(zod_frames=self.zod_frames, frames_id_set=validation_ids, set_type='val', transform=transform_val)

        return train_dataset, validation_set

    def load_datasets(self, num_clients: int):

        train_dataset, validation_set = self.get_zod_datasets()

        # Split training set into `num_clients` partitions to simulate different local datasets
        partition_size = len(train_dataset) // num_clients

        lengths = [partition_size]
        if num_clients > 1:
            lengths = [partition_size] * (num_clients - 1)
            lengths.append(len(train_dataset) - sum(lengths))

        datasets = random_split(train_dataset, lengths, torch.Generator().manual_seed(SEED))

        # Split each partition into train/val and create DataLoader
        # TODO: make loaders as tuple of labeled and unlabeled
        trainloaders, valloaders = [], []
        lengths_train, lengths_val = [], []
        for ds in datasets:
            len_val = int(len(ds) * VAL_FACTOR)
            len_train = int(len(ds) - len_val)
            lengths_train.append(len_train)
            lengths_val.append(len_val)
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(SEED))
            trainloaders.append(DataLoader(ds_train,batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory= True))
            valloaders.append(DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory= True))

        completeTrainloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=True, prefetch_factor=PREFETCH_FACTOR, pin_memory= True)
        completeValloader = DataLoader(validation_set, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=False, prefetch_factor=PREFETCH_FACTOR, pin_memory= True)
        testloader = DataLoader(validation_set, batch_size=len(validation_set), num_workers=NUM_WORKERS, shuffle=False, prefetch_factor=PREFETCH_FACTOR, pin_memory= True)

        """report to tensor board"""
        save_dataset_tb_plot(self.tb_path, lengths_train, "training", SEED)
        save_dataset_tb_plot(self.tb_path, lengths_val, "validation", SEED)

        return (
            trainloaders,
            valloaders,
            testloader,
            completeTrainloader,
            completeValloader,
        )

class ZodDataset(Dataset):
    def __init__(
        self,
        zod_frames,
        frames_id_set,
        set_type,
        transform,
        transform_fixmatch=None
    ):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform
        self.transform_fixmatch  = transform_fixmatch
        self.set_type = set_type

    def __len__(self):
        return len(self.frames_id_set)
    
    def __getitem__(self, idx):

        frame_idx, isLabeled = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]
        
        image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
            
        polygon_annotations = frame.get_annotation(AnnotationProject.EGO_ROAD)
        mask = polygons_to_binary_mask(polygon_annotations)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self._preprocess_mask(mask)

        sample = dict(image=image, mask=mask)
        
        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        mask = np.expand_dims(mask, 0)

        # transform
        transformed_image = self.transform(image)
        if(self.set_type == 'train'):
            image_u_w, image_u_s = self.transform_fixmatch(image)
            sample["image_u_w"] = image_u_w
            sample["image_u_s"] = image_u_s

        sample["image"] = transformed_image
        sample["mask"] = mask
        sample["isLabeled"] = isLabeled

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == False] = 0.0
        mask[mask == True] = 1.0
        return mask


def load_ego_road(dataset_root):
    zod_frames = ZodFrames(dataset_root=dataset_root, version="full")

    loaded_train_seg_annotated_frames = load_from_json(TRAIN_FRAMES_PATH)
    loaded_val_seg_annotated_frames = load_from_json(VAL_FRAMES_PATH)

    print(f'loaded {len(loaded_train_seg_annotated_frames)} train frame ids.')
    print(f'loaded {len(loaded_val_seg_annotated_frames)} val frame ids.')

    return zod_frames, loaded_train_seg_annotated_frames, loaded_val_seg_annotated_frames

def get_frame_seg_mask(zod_frames, frame_id):
    zod_frame = zod_frames[frame_id]

    mask = None
    try:
        # get the ego road annotations
        polygon_annotations = zod_frame.get_annotation(AnnotationProject.EGO_ROAD)

        # convert the polygons to a binary mask (which can be used
        mask = polygons_to_binary_mask(polygon_annotations)
    except:
        print(f'problem with frame {frame_id}')
        
    return mask

def save_to_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))
        
def load_from_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_dataset_tb_plot(tb_path, sample_distribution, subtitle, seed):
    plt.bar(list(range(1, len(sample_distribution) + 1)), sample_distribution)
    plt.xlabel("Partitions")
    plt.ylabel("Samples")
    plt.suptitle("Distribution of samples")
    plt.title("%s, seed: %s" % (subtitle, seed)),

    """report to tensor board"""
    writer = SummaryWriter(tb_path)
    writer.add_figure("sample_distribution/%s" % (subtitle), plt.gcf(), global_step=0)
    writer.close()

def dataset_visualize(train_dataset, valid_dataset, test_dataset):
    # lets look at some samples
    sample = train_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = valid_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = test_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()

def predict_visualize(model, test_dataloader):
    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()