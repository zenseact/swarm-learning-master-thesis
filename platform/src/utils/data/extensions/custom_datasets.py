# Description: Balanced frames generator
# Must be executed as a script and must return two list of frames, train and validation

import os
import numpy as np

from pathlib import Path

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def overfit_turn():
    return ["012589"], ["012589"]

def balanced_frames():
    with open(Path(current_dir, "resources/balanced_train_ids.txt"), "r") as f:
        content_train = f.read().splitlines()

    with open(Path(current_dir, "resources/balanced_val_ids.txt"), "r") as f:
        content_val = f.read().splitlines()

    return content_train, content_val

def interpolated_target_distances(dataset_class: "ZodDataset", idx: int):
    
    def euclidean_distance(coords):
        """
        Calculate the Euclidean distance between successive rows of a given array of coordinates.
        """
        diffs = np.diff(coords, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        return dists
    
    def get_points_at_distance(points, target_distances):
        dists = euclidean_distance(points)
        dists = np.insert(dists, 0, 0) # so that there is a dist for all points in points.
        accumulated_distances = np.cumsum(dists)
        
        interpolated_points = np.empty((len(target_distances), points.shape[1]))
        
        if max(target_distances) > accumulated_distances[-1]:
            raise ValueError("Target distance is larger than the accumulated distance")
        
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
    
    # get the oxts points for the frame
    image, car_relative_points = dataset_class.id_to_car_points(idx)
    
    # Multimodal Trajectory Prediction for Self-driving Vehicles using a Single Monocular Camera
    # Hlib Kilichenko (2023)
    # Subsection 3.1.1, Singular Trajectory Prediction
    target_distances = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
    
    # Get the points at the target distances
    interpolated_car_relative_points = get_points_at_distance(car_relative_points, target_distances)

    label = interpolated_car_relative_points.flatten().astype("float32")
    image = image.astype("float32")

    return label, image
