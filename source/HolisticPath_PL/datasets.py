from gt_utils import *


class ZODImporter:
    def __init__(
        self,
        root=c('dataset_root'),
        subset_factor=c('subset_factor'),
        img_size=c('image_size'),
        batch_size=c('batch_size'),
        tb_path=TB_PATH,
        zod_frames=None,
        training_frames=None, 
        validation_frames=None
    ):
        if(zod_frames == None):
            self.zod_frames = ZodFrames(dataset_root=root, version='full')

            self.training_frames_all = self.zod_frames.get_split(constants.TRAIN)
            self.validation_frames_all = self.zod_frames.get_split(constants.VAL)
            
            self.training_frames, self.validation_frames = self.get_train_val_ids(
                self.training_frames_all, 
                self.validation_frames_all, 
                subset_factor)
        else:
            self.zod_frames = zod_frames
            self.training_frames = training_frames
            self.validation_frames = validation_frames
            
        print("length of training_frames subset:", len(self.training_frames))
        print("length of validation_frames subset:", len(self.validation_frames))

        self.img_size = img_size
        self.batch_size = batch_size
        self.tb_path = tb_path
    
    def get_train_val_ids(self, training_frames_all, validation_frames_all, subset_factor):
        if(c('dataset_division') == 'balanced'):
            with open("balanced_train_ids.txt") as f:
                training_frames_all = json.load(f)
                print(f'balanced sample: {training_frames_all[:5]}')

        if(c('validation_set') == 'kil'):
            validation_frames_all = KIL_VAL_FRAMES
            print(f'Kil sample: {validation_frames_all[:5]}')

        training_frames = list(training_frames_all)[: int(len(training_frames_all) * subset_factor)]
        validation_frames = list(validation_frames_all)[: int(len(validation_frames_all) * subset_factor)]

        training_frames = [x for x in tqdm(training_frames) if self.is_valid(x)]

        if(c('validation_set') != 'kil'):
            validation_frames = [x for x in tqdm(validation_frames) if self.is_valid(x)]

        inter = set(training_frames).intersection(set(validation_frames))
        print('intersection', inter)
        print('len(intersection)', len(inter))
        
        return training_frames, validation_frames
        
    def is_valid(self, frame_id):
        return (str(frame_id).zfill(6) not in UNUSED_FRAMES) and (str(frame_id).zfill(6) not in set(KIL_VAL_FRAMES))
    
    def load_seperate_datasets(self):
        seed = 42

        imagenet_mean=[0.485, 0.456, 0.406]
        imagenet_std=[0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.Resize(size=(self.img_size, self.img_size), antialias=True)
        ])

        trainset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.training_frames, transform=transform)
        valset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.validation_frames, transform=transform)
        testset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.validation_frames, transform=transform)

        trainloaders, valloaders = [], []
        if(c('type')=='federated'):
            # Split training set into `num_clients` partitions to simulate different local datasets
            partition_size = len(trainset) // c('num_clients')

            lengths = [partition_size]
            if c('num_clients') > 1:
                lengths = [partition_size] * (c('num_clients') - 1)
                lengths.append(len(trainset) - sum(lengths))

            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

            # Split each partition into train/val and create DataLoader
            for ds in datasets:
                trainloaders.append(DataLoader(ds,batch_size=self.batch_size, shuffle=True, num_workers=c('num_workers')))
                valloaders.append(DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=c('num_workers')))


        completeTrainloader = DataLoader(
            trainset, batch_size=self.batch_size, num_workers=c('num_workers'), shuffle=True, 
            prefetch_factor=c('prefetch_factor'),
            pin_memory= True)
        
        completeValloader = DataLoader(
            valset, batch_size=self.batch_size, num_workers=c('num_workers'), shuffle=False,
            prefetch_factor=c('prefetch_factor'),
            pin_memory= True)

        testloader = DataLoader(testset, batch_size=len(self.validation_frames), shuffle=False, num_workers=c('num_workers'))

        print("length of training_frames subset:", len(self.training_frames))
        print("length of validation_frames subset:", len(self.validation_frames))
        print("length of testing_frames subset:", len(self.validation_frames))
        inter = set(self.training_frames).intersection(set(self.validation_frames))
        print("intersection train-val:", inter)
        print("len:", len(inter))

        return (
            trainloaders, 
            valloaders,
            testloader,
            completeTrainloader,
            completeValloader,
        )
        
    def load_datasets(self, num_clients=c('num_clients')):
        if(c('validation_set') == 'kil'):
            return self.load_seperate_datasets()

        seed = 42
        imagenet_mean=[0.485, 0.456, 0.406]
        imagenet_std=[0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.Resize(size=(self.img_size, self.img_size), antialias=True)
        ])

        trainset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.training_frames, transform=transform)
        testset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.validation_frames, transform=transform)

        # Split training set into `num_clients` partitions to simulate different local datasets
        partition_size = len(trainset) // num_clients

        lengths = [partition_size]
        if num_clients > 1:
            lengths = [partition_size] * (num_clients - 1)
            lengths.append(len(trainset) - sum(lengths))

        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

        # Split each partition into train/val and create DataLoader
        trainloaders, valloaders = [], []
        lengths_train, lengths_val = [], []
        for ds in datasets:
            len_val = int(len(ds) * c('val_factor'))
            len_train = int(len(ds) - len_val)
            lengths_train.append(len_train)
            lengths_val.append(len_val)
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train,batch_size=self.batch_size, shuffle=True, num_workers=c('num_workers')))
            valloaders.append(DataLoader(ds_val, batch_size=self.batch_size, num_workers=c('num_workers')))

        len_complete_val = int(len(trainset) * c('val_factor'))
        len_complete_train = int(len(trainset) - len_complete_val)
        train_split, val_split = random_split(
            trainset,
            [len_complete_train, len_complete_val],
            torch.Generator().manual_seed(seed),
        )

        completeTrainloader = DataLoader(
            train_split, batch_size=self.batch_size, num_workers=c('num_workers'), shuffle=True, 
            prefetch_factor=c('prefetch_factor'),
            pin_memory= True)
        
        completeValloader = DataLoader(
            val_split, batch_size=self.batch_size, num_workers=c('num_workers'), shuffle=True,
            prefetch_factor=c('prefetch_factor'),
            pin_memory= True)

        testloader = DataLoader(testset, batch_size=self.batch_size, num_workers=c('num_workers'))

        """report to tensor board"""
        save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)
        save_dataset_tb_plot(self.tb_path, [len(testset)], "testing", seed)

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
        transform=None,
        target_transform=None,
    ):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.frames_id_set)
    
    def __getitem__(self, idx):
        
        # load frame
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]
        
        # get image
        image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # extract ground truth
        label = get_ground_truth(self.zod_frames, frame_idx)
        
        # create sample
        sample = dict(image=image, label=label)
        
        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((c('image_size'), c('image_size')), Image.BILINEAR))
        
        # convert to other format HWC -> CHW
        #sample["image"] = np.moveaxis(image, -1, 0)
        sample["label"] = np.expand_dims(label, 0).astype(np.float32)
        
        if(self.transform):
            sample["image"] = self.transform(sample["image"])
        
        return sample
