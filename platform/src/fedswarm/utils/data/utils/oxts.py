from matplotlib import pyplot as plt
import numpy as np
import torch

from cv2 import circle
from zod.constants import Camera
from zod.utils.geometry import (
    get_points_in_camera_fov,
    project_3d_to_2d_kannala,
    transform_points,
)


def id_to_car_points(zod_frames, frame_id, return_image=True):
    frame = zod_frames[frame_id]
    # get image
    if return_image:
        image = frame.get_image()
    else:
        image = None
    # extract oxts
    oxts = frame.oxts
    # get timestamp
    key_timestamp = frame.info.keyframe_time.timestamp()
    # get posses associated with frame timestamp
    current_pose = oxts.get_poses(key_timestamp)
    # transform the points to the car coordinate system
    transformed_poses = np.linalg.pinv(current_pose) @ oxts.poses
    points = transformed_poses[:, :3, -1]
    points = points[points[:, 0] > 0]
    return image, points


def car_points_to_camera(zod_frames, frame_id, car_points):
    camera = Camera.FRONT
    frame = zod_frames[frame_id]
    calibs = frame.calibration
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camera_points = transform_points(car_points[:, :3], T_inv)
    return camera_points


def camera_points_to_2d(zod_frames, frame_id, camera_points):
    camera = Camera.FRONT
    frame = zod_frames[frame_id]
    calibs = frame.calibration
    # filter points that are not in the camera field of view
    points_in_fov, _ = get_points_in_camera_fov(
        calibs.cameras[camera].field_of_view, camera_points
    )

    # project points to image plane
    image_points = project_3d_to_2d_kannala(
        points_in_fov,
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )
    return image_points


def get_image(zod_frames, frame_id):
    return zod_frames[frame_id].get_image()


def draw_circles(image, image_points, colour=(255, 0, 0)):
    for i in range(image_points.shape[0]):
        x, y = int(image_points[i, 0]), int(image_points[i, 1])
        circle(image, (x, y), 20, colour, -1)
    return image


def visualize_car_points(
    zod_frames, frame_id, car_points, colour=(255, 0, 0), image=None
):
    camera_points = car_points_to_camera(zod_frames, frame_id, car_points)
    image_points = camera_points_to_2d(zod_frames, frame_id, camera_points)
    if image is None:
        image = get_image(zod_frames, frame_id)
    image = draw_circles(image, image_points, colour)
    return image


def plot_predictions(model, zod_frames, dataset, idx):
    # Define the device to be used (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the input to the model
    input_tensor, ground_truth = dataset[idx]
    frame_id = dataset.frames_id_set[idx]

    # Move the input tensor to the specified device
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Disable autograd to speed up inference
    with torch.no_grad():
        # Pass the input through the model to get the output
        output = model(input_tensor)

    # Convert the output to a numpy array
    output = output.cpu().numpy()
    output = output.reshape(17, 3)
    ground_truth = ground_truth.reshape(17, 3)

    # image_with_ground_truth = visualize_car_points(zod_frames, frame_id, ground_truth, colour=(255, 0, 0))
    image_with_gt = visualize_car_points(
        zod_frames, frame_id, ground_truth, colour=(0, 0, 255)
    )
    image_with_prediction = visualize_car_points(
        zod_frames, frame_id, output, colour=(255, 0, 0), image=image_with_gt
    )

    # TODO: use config loss metric and not hardcoded L1
    loss = torch.nn.L1Loss()(torch.tensor(output).squeeze(), torch.tensor(ground_truth))
    return image_with_prediction, loss.item(), ground_truth, output


def present_predictions(img, loss, ground_truth, output, frame_id):
    plt.subplot(1, 6, (1, 3))
    plt.imshow(img)
    plt.title("Camera view")

    plt.subplot(1, 6, (4, 6))
    plt.scatter(x=ground_truth[:, 1], y=ground_truth[:, 0], s=1, c="blue")
    plt.scatter(x=output[:, 1], y=output[:, 0], s=1, c="red")
    largest_x = max(abs(min(ground_truth[:, 1])), abs(max(ground_truth[:, 1])))
    x_threshold = 20
    lower_x_limit = -max(largest_x, x_threshold)
    upper_x_limit = max(largest_x, x_threshold)
    plt.xlim(upper_x_limit, lower_x_limit)
    plt.legend(["Ground Truth", "Prediction"])
    plt.title("Birds-eye view")

    plt.suptitle(
        "{} | Prediction vs Ground Truth | loss: {}".format(frame_id, loss))


def euclidean_distance(coords):
    """
    Calculate the Euclidean distance between successive rows of a given array of coordinates.
    """
    diffs = np.diff(coords, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return dists


def get_points_at_distance(points, target_distances):
    dists = euclidean_distance(points)
    # so that there is a dist for all points in points.
    dists = np.insert(dists, 0, 0)
    accumulated_distances = np.cumsum(dists)

    interpolated_points = np.empty((len(target_distances), points.shape[1]))

    if max(target_distances) > accumulated_distances[-1]:
        raise ValueError(
            "Target distance is larger than the accumulated distance")

    index = 0
    inter_idx = 0
    for target_distance in target_distances:
        # Increment index until we have passed the target distance
        while accumulated_distances[index] < target_distance:
            index += 1
        # If we reach this state, then index - 1 is the closest index before going over.
        # Check if the target distance is exactly at a point in the list
        if accumulated_distances[index - 1] == target_distance:
            interpolated_points[inter_idx] = points[index - 1]
            inter_idx += 1
        else:
            # Interpolate between the two nearest points
            p1 = points[index - 1]
            p2 = points[index]
            d1 = accumulated_distances[index - 1]
            d2 = accumulated_distances[index]
            t = (target_distance - d1) / (d2 - d1)
            interpolated_points[inter_idx] = p1 + t * (p2 - p1)
            inter_idx += 1
    return interpolated_points
