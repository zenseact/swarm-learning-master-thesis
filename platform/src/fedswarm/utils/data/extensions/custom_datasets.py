# Description: Balanced frames generator
# Must be executed as a script and must return two list of frames, train and validation

import os
import numpy as np

from pathlib import Path
from fedswarm.utils.data.utils.oxts import get_points_at_distance, id_to_car_points
from zod.constants import AnnotationProject

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def overfit_turn():
    return ["012589"], ["012589"]


def balanced_frames():
    with open(Path(current_dir, "resources/balanced_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/balanced_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val


def manually_selected():
    with open(Path(current_dir, "resources/manually_selected_train.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/manually_selected_val.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val


def balanced_frames_borrowed(train=0.9):
    with open(Path(current_dir, "resources/balanced_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/balanced_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()

    all_samples = content_train + content_val
    # We assume here that the train and val sets are shuffled already to avoid bias
    index_threshold = int(len(all_samples) * train)

    return all_samples[:index_threshold], all_samples[index_threshold:]


def turns(direction: str = "right"):
    assert direction in ["right", "left"]
    with open(Path(current_dir, f"resources/train_turns_{direction}.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, f"resources/val_turns_{direction}.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val

# GET ITEM FUNCTIONS FOR THE DATASET


def interpolated_target_distances(dataset_class: "ZodDataset", idx: int):
    zod_frames = dataset_class.zod_frames
    frame_id = dataset_class.frames_id_set[idx]

    # get the oxts points for the frame
    image, car_relative_points = id_to_car_points(zod_frames, frame_id)

    # Multimodal Trajectory Prediction for Self-driving Vehicles using a Single Monocular Camera
    # Hlib Kilichenko (2023)
    # Subsection 3.1.1, Singular Trajectory Prediction
    target_distances = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        50,
        60,
        70,
        80,
        95,
        110,
        125,
        145,
        165,
    ]

    # Get the points at the target distances
    interpolated_car_relative_points = get_points_at_distance(
        car_relative_points, target_distances
    )

    label = interpolated_car_relative_points.flatten().astype("float32")
    image = image.astype("uint8")

    return label, image


def interpolated_target_distances_short(dataset_class: "ZodDataset", idx: int):
    zod_frames = dataset_class.zod_frames
    frame_id = dataset_class.frames_id_set[idx]

    # get the oxts points for the frame
    image, car_relative_points = id_to_car_points(zod_frames, frame_id)

    target_distances = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
    ]

    # Get the points at the target distances
    interpolated_car_relative_points = get_points_at_distance(
        car_relative_points, target_distances
    )

    label = interpolated_car_relative_points.flatten().astype("float32")
    image = image.astype("uint8")

    return label, image


def ego_road_segmentation(dataset_class: "ZodDataset", idx: int):
    zod_frames = dataset_class.zod_frames
    frame_id = dataset_class.frames_id_set[idx]

    print(f"GetItem: [idx: {idx}, {frame_id}]")

    image = zod_frames[frame_id].get_image().astype("uint8")

    label = zod_frames[frame_id].get_annotation(AnnotationProject.EGO_ROAD)

    print(
        f"GetItem: [idx: {idx}, {frame_id}] ImageType {type(image)}, LabelType {type(label)}, LabelLen {len(label)}")

    return label, image


def noise(dataset_class: "ZodDataset", idx: int):
    zod_frames = dataset_class.zod_frames
    frame_id = dataset_class.frames_id_set[idx]

    # get the oxts points for the frame
    _, car_relative_points = id_to_car_points(
        zod_frames, frame_id, return_image=False)

    # Multimodal Trajectory Prediction for Self-driving Vehicles using a Single Monocular Camera
    # Hlib Kilichenko (2023)
    # Subsection 3.1.1, Singular Trajectory Prediction
    target_distances = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        50,
        60,
        70,
        80,
        95,
        110,
        125,
        145,
        165,
    ]

    # Get the points at the target distances
    interpolated_car_relative_points = get_points_at_distance(
        car_relative_points, target_distances
    )

    # Define dimensions of the image
    # source img = (2168, 3848, 3)
    x = 3848
    y = 2168

    # Generate random RGB noise image
    image = np.random.randint(0, 256, (y, x, 3), dtype=np.uint8)

    label = interpolated_car_relative_points.flatten().astype("float32")
    image = image.astype("uint8")

    return label, image
