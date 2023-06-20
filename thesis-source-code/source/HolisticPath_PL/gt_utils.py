
from static_params import *

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
        pose_idx = [accumulated_distances.index(i) for i in c('target_distances')]
        used_poses = transformed_poses[pose_idx]

    except:
        print("detected invalid frame: ", frame_id)
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
    #writer.add_figure("sample_distribution/%s" % (subtitle), plt.gcf(), global_step=0)


def reshape_ground_truth(label, output_size=c('output_size')):
    return label.reshape(((c('output_size') // 3), 3))

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
    return image, points_gt, preds_row
    
def visualize_multiple(zod_frames, frame_ids, model=None):
    if(model):
        images = [visualize_HP_on_image(zod_frames, frame_id, predict(model.to(DEVICE), zod_frames, frame_id), showImg=False) for frame_id in frame_ids]
    else:
        images = [visualize_HP_on_image(zod_frames, frame_id, None, showImg=False) for frame_id in frame_ids]
        
    plt.figure(figsize=(60,60))
    columns = 4
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, image in enumerate(images):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.gca().set_title(frame_ids[i], fontsize=20)
        plt.imshow(image[0])  
    
def get_transformed_image(zod_frames, frame_id):
    frame = zod_frames[frame_id]
    image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
    image = np.array(Image.open(image_path).convert("RGB"))
    image = np.array(Image.fromarray(image).resize((c('image_size'), c('image_size')), Image.BILINEAR))
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(c('image_size'), c('image_size')), antialias=True)
        ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image

def predict(model, zod_frames, frame_id):
    image = get_transformed_image(zod_frames, frame_id).to(DEVICE)
    outputs = model(image).squeeze(0)
    label = torch.from_numpy(get_ground_truth(zod_frames, frame_id)).to(DEVICE)
    
    loss = torch.nn.L1Loss(label, outputs)

    print(f'frame: {frame_id}, loss: {loss.item()}')
    preds = outputs.cpu().detach().numpy()
    return preds

def load_HP(dataset_root):
    zod_frames = ZodFrames(dataset_root=dataset_root, version='full')
    training_frames_all = zod_frames.get_split(constants.TRAIN)
    validation_frames_all = zod_frames.get_split(constants.VAL)

    return zod_frames, training_frames_all, validation_frames_all

def is_valid(frame_id):
    return frame_id not in UNUSED_FRAMES

def save_to_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))
        
def load_from_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    
