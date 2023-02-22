from static_params import *
from utilities import *

class ZODImporter:
    def __init__(self, root=None, subset_factor=SUBSET_FACTOR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, tb_path=None):
        dataset_root = root if root else "/staging/dataset_donation/round_2"
        version = "full"  # "mini" or "full"
        self.zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

        training_frames_all = self.zod_frames.get_split(constants.TRAIN)
        validation_frames_all = self.zod_frames.get_split(constants.VAL)

        self.training_frames = training_frames_all[:int(len(training_frames_all) * subset_factor)]
        self.validation_frames = validation_frames_all[:int(len(validation_frames_all) * subset_factor)]

        print('training_frames length:', len(self.training_frames))
        print('validation_frames length:', len(self.validation_frames))
        self.img_size = img_size
        self.batch_size = batch_size
        self.tb_path = tb_path

    def load_datasets(self, num_clients: int):
        seed = 42
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.img_size, self.img_size))])

        trainset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.training_frames, transform=transform)
        testset = ZodDataset(zod_frames=self.zod_frames, frames_id_set=self.validation_frames, transform=transform)

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
            trainloaders.append(DataLoader(ds_train, batch_size=self.batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.batch_size))
        testloader = DataLoader(testset, batch_size=self.batch_size)

        """report to tensor board"""
        save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

        return trainloaders, valloaders, testloader


class ZodDataset(Dataset):
    def __init__(self, zod_frames, frames_id_set, transform=None, target_transform=None):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]

        image = frame.get_image(Anonymization.DNAT)
        poses = frame.ego_motion.poses
        x = poses[:, 0:1, 3]
        y = poses[:, 1:2, 3]
        z = poses[:, 2:3, 3]
        coordinates = np.append(x, y)
        coordinates = np.append(coordinates, z)

        label = coordinates.astype('float32')
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
    trainloaders, valloaders, testloader = zod.load_datasets(nr_clients)

    print('nr of training imgs:', len(trainloaders[0].dataset))
    print('nr of validation imgs:', len(valloaders[0].dataset))
    print('nr of test imgs:', len(testloader.dataset))
    print('input shape:', trainloaders[0].dataset[0][0].shape)
    print('output shape:', trainloaders[0].dataset[0][1].shape)


if __name__ == "__main__":
    main()
