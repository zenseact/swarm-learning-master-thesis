from static_params import *

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
            self.zod_frames = ZodFrames(dataset_root=root, version='full')
            self.training_frames_all = self.zod_frames.get_split(constants.TRAIN)
            self.validation_frames_all = self.zod_frames.get_split(constants.VAL)
            
            self.training_frames = list(self.training_frames_all)[: int(len(self.training_frames_all) * subset_factor)]
            self.validation_frames = list(self.validation_frames_all)[: int(len(self.validation_frames_all) * subset_factor)]

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
        return frame_id not in UNUSED_FRAMES and get_ground_truth(self.zod_frames, frame_id).shape == (NUM_OUTPUT,)
        
    def load_datasets(self, num_clients: int):
        seed = 42
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(FRAMES_IMAGE_MEAN, FRAMES_IMAGE_STD),
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
            len_val = int(len(ds) * VAL_FACTOR)
            len_train = int(len(ds) - len_val)
            lengths_train.append(len_train)
            lengths_val.append(len_val)
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train,batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS))
            valloaders.append(DataLoader(ds_val, batch_size=self.batch_size, num_workers=NUM_WORKERS))

        len_complete_val = int(len(trainset) * VAL_FACTOR)
        len_complete_train = int(len(trainset) - len_complete_val)
        train_split, val_split = random_split(
            trainset,
            [len_complete_train, len_complete_val],
            torch.Generator().manual_seed(seed),
        )

        completeTrainloader = DataLoader(train_split, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        completeValloader = DataLoader(val_split, batch_size=self.batch_size, num_workers=NUM_WORKERS)

        testloader = DataLoader(testset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

        """report to tensor board"""
        save_dataset_tb_plot(self.tb_path, lengths_train, "training", seed)
        save_dataset_tb_plot(self.tb_path, lengths_val, "validation", seed)

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
        image = np.array(Image.fromarray(sample["image"]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["label"] = np.expand_dims(label, 0)
        
        return sample

def get_ground_truth(zod_frames, frame_id):
    # get frame
    zod_frame = zod_frames[frame_id]

    # extract oxts
    oxts = zod_frame.oxts

    # get timestamp
    key_timestamp = zod_frame.info.keyframe_time.timestamp()

    try:
        # get posses associated with frame timestamp
        current_pose = oxts.get_poses(key_timestamp)

        # transform poses
        all_poses = oxts.poses[oxts.timestamps>=key_timestamp]
        transformed_poses = np.linalg.pinv(current_pose) @ all_poses

        # get translations
        translations = transformed_poses[:, :3, 3]

        # calculate acc diff distance
        distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        accumulated_distances = np.cumsum(distances).astype(int).tolist()

        # get the poses that each have a point having a distance from TARGET_DISTANCES
        pose_idx = [accumulated_distances.index(i) for i in TARGET_DISTANCES]
        used_poses = transformed_poses[pose_idx]

    except:
        #print("detected invalid frame: ", frame_id)
        return np.array([])

    #print(used_poses.shape)
    points = used_poses[:, :3, -1]
    return points.flatten()

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


def reshape_ground_truth(label, output_size=NUM_OUTPUT):
    return label.reshape(((NUM_OUTPUT // 3), 3))

def visualize_HP_on_image(zod_frames, frame_id, preds=None, showImg=True):
    """Visualize oxts track on image plane."""
    camera=Camera.FRONT
    zod_frame = zod_frames[frame_id]
    image = zod_frame.get_image(Anonymization.DNAT)
    calibs = zod_frame.calibration
    points_gt = get_ground_truth(zod_frames, frame_id)
    preds_row = preds.copy()
    points = reshape_ground_truth(points_gt)
    
    circle_size = 15
    
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camerapoints = transform_points(points[:, :3], T_inv)

    # filter points that are not in the camera field of view
    points_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, camerapoints)
    points_in_fov = points_in_fov[0]

    # project points to image plane
    xy_array = project_3d_to_2d_kannala(
        points_in_fov,
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )
    
    ground_truth_color = (19, 80, 41)
    preds_color = (161, 65, 137)
    
    points = []
    for i in range(xy_array.shape[0]):
        x, y = int(xy_array[i, 0]), int(xy_array[i, 1])
        cv2.circle(image, (x,y), circle_size, ground_truth_color, -1)
        points.append([x,y])
    
    """Draw a line in image."""
    def draw_line(image, line, color):
        return cv2.polylines(image.copy(), [np.round(line).astype(np.int32)], isClosed=False, color=color, thickness=20)
    
    image = draw_line(image, points, ground_truth_color)
    
    # transform and draw predictions 
    if(preds is not None):
        preds = reshape_ground_truth(preds)
        predpoints = transform_points(preds[:, :3], T_inv)
        predpoints_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, predpoints)
        predpoints_in_fov = predpoints_in_fov[0]
        
        xy_array_preds = project_3d_to_2d_kannala(
            predpoints_in_fov,
            calibs.cameras[camera].intrinsics[..., :3],
            calibs.cameras[camera].distortion,
        )
        preds = []
        for i in range(xy_array_preds.shape[0]):
            x, y = int(xy_array_preds[i, 0]), int(xy_array_preds[i, 1])
            cv2.circle(image, (x,y), circle_size, preds_color, -1)
            preds.append([x,y])
        
        #preds = preds[:(len(preds)//2)]
        image = draw_line(image, preds, preds_color)
        
    #plt.imsave(f'inference_{frame_id}.png', image)
    if(showImg):
        plt.clf()
        plt.axis("off")
        plt.imshow(image)
    #return image, points_gt, preds_row
    
def get_transformed_image(zod_frames, frame_id):
    frame = zod_frames[frame_id]
    image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
    image = np.array(Image.open(image_path).convert("RGB"))
    image = np.array(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    print(image.shape)
    return image

def predict(model, zod_frames, frame_id):
    image = get_transformed_image(zod_frames, frame_id).to(DEVICE)
    outputs = model(image)
    preds = outputs.cpu().detach().numpy()
    return preds