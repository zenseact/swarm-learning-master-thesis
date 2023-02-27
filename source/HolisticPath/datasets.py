from static_params import *
from utilities import *
from groundtruth_utils import * 

class ZODImporter:
    def __init__(self, root=DATASET_ROOT, subset_factor=SUBSET_FACTOR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, tb_path=None, stored_gt_path=None):
        version = "full"  # "mini" or "full"
        self.zod_frames = ZodFrames(dataset_root=root, version=version)

        training_frames_all = self.zod_frames.get_split(constants.TRAIN)
        validation_frames_all = self.zod_frames.get_split(constants.VAL)

        self.ground_truth = None
        if(stored_gt_path):
            self.ground_truth = load_ground_truth(stored_gt_path)
            print('loaded stored ground truth')

        training_frames_all = [idx for idx in training_frames_all if self.is_valid_frame(idx)]
        validation_frames_all = [idx for idx in validation_frames_all if self.is_valid_frame(idx)]

        self.training_frames = training_frames_all[:int(len(training_frames_all) * subset_factor)]
        self.validation_frames = validation_frames_all[:int(len(validation_frames_all) * subset_factor)]

        print('training_frames length:', len(self.training_frames))
        print('test_frames length:', len(self.validation_frames))
        self.img_size = img_size
        self.batch_size = batch_size
        self.tb_path = tb_path

    def is_valid_frame(self, frame_id):
        if(self.ground_truth):
            return frame_id in self.ground_truth
        else:
            return get_ground_truth(self.zod_frames, frame_id).shape[0] == OUTPUT_SIZE*3

    def load_datasets(self, num_clients: int):
        seed = 42
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.img_size, self.img_size))])

        trainset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.training_frames, stored_ground_truth=self.ground_truth, transform=transform)
        testset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.validation_frames, stored_ground_truth=self.ground_truth, transform=transform)

        # Split training set into `num_clients` partitions to simulate different local datasets
        partition_size = len(trainset) // num_clients
        lengths = [partition_size] * (num_clients - 1)
        lengths.append(len(trainset) - sum(lengths))
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

        # Split each partition into train/val and create DataLoader
        trainloaders, valloaders = [], []
        lengths_train, lengths_val = [], []
        for ds in datasets:
            len_val = int(len(ds) * VAL_FACTOR)
            len_train = int(len(ds) - len_val)
            lengths_train.append(len_train)
            lengths_val.append(len_val)
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, num_workers=10))
            valloaders.append(DataLoader(ds_val, batch_size=self.batch_size, num_workers=10))
        
        len_complete_val = int(len(trainset) * VAL_FACTOR)
        len_complete_train = int(len(trainset) - len_complete_val)
        train_split, val_split = random_split(trainset, [len_complete_train, len_complete_val], torch.Generator().manual_seed(seed))

        completeTrainloader = DataLoader(train_split, batch_size=self.batch_size, num_workers=10)
        completeValloader = DataLoader(val_split, batch_size=self.batch_size, num_workers=10)

        testloader = DataLoader(testset, batch_size=self.batch_size, num_workers=10)

        """report to tensor board"""
        save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

        return trainloaders, valloaders, testloader, completeTrainloader, completeValloader 


class ZodDataset(Dataset):
    def __init__(self, zod_frames, frames_id_set, stored_ground_truth=None, transform=None, target_transform=None):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform
        self.target_transform = target_transform
        self.stored_ground_truth = stored_ground_truth

    def __len__(self):
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]

        image = frame.get_image(Anonymization.DNAT)
        label = None

        if(self.stored_ground_truth):
            label = self.stored_ground_truth[frame_idx]
        else: 
            label = get_ground_truth(self.zod_frames, frame_idx)
        
        label = label.astype('float32')
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def main(
        nr_clients=2,
        subset_factor=SUBSET_FACTOR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE):

    # import Zod data into memory
    zod = ZODImporter(subset_factor=subset_factor, img_size=img_size, batch_size=batch_size)

    # create pytorch loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zod.load_datasets(nr_clients)

    print('nr of training imgs:', len(trainloaders[0].dataset))
    print('nr of validation imgs:', len(valloaders[0].dataset))
    print('nr of test imgs:', len(testloader.dataset))
    print('input shape:', trainloaders[0].dataset[0][0].shape)
    print('output shape:', trainloaders[0].dataset[0][1].shape)


if __name__ == "__main__":
    main()
