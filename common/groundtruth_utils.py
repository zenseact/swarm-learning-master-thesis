from static_params import *
from utilities import * 


def get_ground_truth(zod_frames, frame_id):
    # get frame
    zod_frame = zod_frames[frame_id]
    
    # extract oxts
    oxts = zod_frame.oxts
    
    # get timestamp
    key_timestamp = zod_frame.info.keyframe_time.timestamp()
    
    # get posses associated with frame timestamp
    try:
        current_pose = oxts.get_poses(key_timestamp)
        # transform poses
        all_poses = oxts.poses
        transformed_poses = np.linalg.pinv(current_pose) @ all_poses

        def travelled_distance(poses) -> np.ndarray:
            translations = poses[:, :3, 3]
            distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
            accumulated_distances = np.cumsum(distances).astype(int).tolist()

            pose_idx = [accumulated_distances.index(i) for i in TARGET_DISTANCES] 
            return poses[pose_idx]

        used_poses = travelled_distance(transformed_poses)
    
    except:
        print('detected invalid frame: ', frame_id)
        return np.array([])
    
    print(used_poses.shape)
    points = used_poses[:, :3, -1]
    return flatten_ground_truth(points)

    

def transform_pred(zod_frames, frame_id, pred):
    zod_frame = zod_frames[frame_id]
    key_timestamp = zod_frame.info.keyframe_time.timestamp()
    current_pose = oxts.get_poses(key_timestamp)
    pred = reshape_ground_truth(pred)
    return np.linalg.pinv(current_pose) @ pred


def visualize_HP_on_image(zod_frames, frame_id, preds=None):
    """Visualize oxts track on image plane."""
    camera=Camera.FRONT
    zod_frame = zod_frames[frame_id]
    image = zod_frame.get_image(Anonymization.DNAT)
    calibs = zod_frame.calibration
    points = get_ground_truth(zod_frames, frame_id)
    points = reshape_ground_truth(points)
    
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camerapoints = transform_points(points[:, :3], T_inv)
    print(f"Number of points: {points.shape[0]}")

    # filter points that are not in the camera field of view
    points_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, camerapoints)
    print(f"Number of points in fov: {len(points_in_fov)}")

    # project points to image plane
    xy_array = project_3d_to_2d_kannala(
        points_in_fov,
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )
    
    points = []
    for i in range(xy_array.shape[0]):
        x, y = int(xy_array[i, 0]), int(xy_array[i, 1])
        cv2.circle(image, (x,y), 2, (255, 0, 0), -1)
        points.append([x,y])
    
    """Draw a line in image."""
    def draw_line(image, line, color):
        return cv2.polylines(image.copy(), [np.round(line).astype(np.int32)], isClosed=False, color=color, thickness=10)
    
    ground_truth_color = (19, 80, 41)
    preds_color = (161, 65, 137)
    image = draw_line(image, points, ground_truth_color)
    
    # transform and draw predictions 
    if(preds):
        preds = reshape_ground_truth(preds)
        print(f"Number of pred points on image: {preds.shape[0]}")
        predpoints = transform_points(preds[:, :3], T_inv)
        predpoints_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, predpoints)
        xy_array_preds = project_3d_to_2d_kannala(
            predpoints_in_fov,
            calibs.cameras[camera].intrinsics[..., :3],
            calibs.cameras[camera].distortion,
        )
        preds = []
        for i in range(xy_array_preds.shape[0]):
            x, y = int(xy_array_preds[i, 0]), int(xy_array_preds[i, 1])
            cv2.circle(image, (x,y), 2, (255, 0, 0), -1)
            preds.append([x,y])
        image = draw_line(image, preds, preds_color)
        
    plt.clf()
    plt.axis("off")
    plt.imsave(f'inference_{frame_id}.png', image)
    #plt.imshow(image)

def flatten_ground_truth(label):
    return label.flatten()

def reshape_ground_truth(label, output_size=NUM_OUTPUT):
    return label.reshape(((NUM_OUTPUT//3),3))

def create_ground_truth(zod_frames, training_frames, validation_frames, path):
    all_frames = validation_frames.copy()
    all_frames.extend(training_frames)
    
    corrupted_frames = []
    ground_truth = {}
    for frame_id in tqdm(all_frames):
        gt = get_ground_truth(zod_frames, frame_id)
        if(gt.shape[0] != NUM_OUTPUT):
            corrupted_frames.append(frame_id)
            continue
        else:
            ground_truth[frame_id] = gt.tolist()
        
    # Serializing json
    json_object = json.dumps(ground_truth, indent=4)

    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)
    
    print(corrupted_frames)

def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt




def main():
    zod_frames = ZodFrames(dataset_root=DATASET_ROOT, version='full')
    training_frames_all = zod_frames.get_split(constants.TRAIN)
    validation_frames_all = zod_frames.get_split(constants.VAL)

    #idx = "081294"
    #image = visualize_HP_on_image(zod.zod_frames, idx)

    create_ground_truth(zod_frames, training_frames_all, validation_frames_all, STORED_GROUND_TRUTH_PATH)

if __name__ == "__main__":
    main()