# Description: Balanced frames generator
# Must be executed as a script and must return two list of frames, train and validation

import os
import numpy as np

from pathlib import Path
from fedswarm.utils.data.utils.oxts import get_points_at_distance, id_to_car_points

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def overfit_turn():
    return ["012589"], ["012589"]

def balanced_frames():
    with open(Path(current_dir, "resources/balanced_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/balanced_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val

def balanced_frames_borrowed(train=0.9):
    with open(Path(current_dir, "resources/balanced_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/balanced_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()
    
    all_samples =content_train + content_val
    # We assume here that the train and val sets are shuffled already to avoid bias
    index_threshold = int(len(all_samples) * train)
    
    return all_samples[:index_threshold], all_samples[index_threshold:]

def turns(direction: str = "right"):
    assert direction in ["right", "left"]
    with open(Path(current_dir, f"resources/{direction}_turns_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, f"resources/{direction}_turns_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val

def interpolated_target_distances(dataset_class: "ZodDataset", idx: int):
    
    zod_frames = dataset_class.zod_frames
    frame_id = dataset_class.frames_id_set[idx]
    
    # get the oxts points for the frame
    image, car_relative_points = id_to_car_points(zod_frames, frame_id)
    
    # Multimodal Trajectory Prediction for Self-driving Vehicles using a Single Monocular Camera
    # Hlib Kilichenko (2023)
    # Subsection 3.1.1, Singular Trajectory Prediction
    target_distances = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
    
    # Get the points at the target distances
    interpolated_car_relative_points = get_points_at_distance(car_relative_points, target_distances)

    label = interpolated_car_relative_points.flatten().astype("float32")
    image = image.astype("float32")

    return label, image
