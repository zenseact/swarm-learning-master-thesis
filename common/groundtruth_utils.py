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
    current_pose = oxts.get_poses(key_timestamp)
    
    # transform poses
    all_poses = oxts.poses
    transformed_poses = np.linalg.pinv(current_pose) @ all_poses
    points = transformed_poses[:, :3, -1]
    points = points[points[:, 0] > 0]    
    
    # get equally distributed points 
    nr_points = points.shape[0] // 2
    points = np.array([points[i] for i in range(0,nr_points,nr_points//OUTPUT_SIZE)][:OUTPUT_SIZE])
    return flatten_ground_truth(points)
    
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

def reshape_ground_truth(label, output_size=OUTPUT_SIZE):
    return label.reshape((output_size,3))

def create_ground_truth(zod_frames, training_frames, validation_frames, path):
    all_frames = validation_frames.copy()
    all_frames.extend(training_frames)
    
    ground_truth = {}
    for frame_id in tqdm(all_frames):
        gt = visualize_HP_on_image(zod_frames, frame_id)
        if(gt.shape[0] != OUTPUT_SIZE*3):
            print('detected invalid frame: ', frame_id)
            continue
        else:
            ground_truth[frame_id] = gt.tolist()
        
    # Serializing json
    json_object = json.dumps(ground_truth, indent=4)

    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)

def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt




def main():
    from datasets import ZODImporter
    
    zod = ZODImporter(subset_factor=SUBSET_FACTOR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, stored_gt_path=STORED_GROUND_TRUTH_PATH)

    idx = "081294"
    image = visualize_HP_on_image(zod.zod_frames, idx)


if __name__ == "__main__":
    main()