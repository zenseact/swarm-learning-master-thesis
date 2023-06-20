from zod import ZodFrames
import zod.constants as constants
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
from zod.data_classes import LidarData
from zod.utils.polygon_transformations import polygons_to_binary_mask
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

TRAIN_FRAMES_PATH = "GroundTruth/training_seg_annotated_frames.json"
VAL_FRAMES_PATH = "GroundTruth/validation_seg_annotated_frames.json"

def load_zod():
    dataset_root = "/mnt/ZOD"  # AI Sweden
    version = "full"  # "mini" or "full"

    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    # get default training and validation splits
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    return zod_frames, training_frames, validation_frames

def load_ego_road(dataset_root):
    zod_frames = ZodFrames(dataset_root=dataset_root, version="full")

    loaded_train_seg_annotated_frames = load_from_json(TRAIN_FRAMES_PATH)
    loaded_val_seg_annotated_frames = load_from_json(VAL_FRAMES_PATH)

    print(f'loaded {len(loaded_train_seg_annotated_frames)} train frame ids.')
    print(f'loaded {len(loaded_val_seg_annotated_frames)} val frame ids.')

    return zod_frames, loaded_train_seg_annotated_frames, loaded_val_seg_annotated_frames

def get_seg_annotated_frame_ids(zod_frames, frame_ids):
    seg_annotated_frames = []
    for frame_id in tqdm(frame_ids):
        zod_frame = zod_frames[frame_id]
        if(len(zod_frame.info.annotation_frames[AnnotationProject.EGO_ROAD]) == 0):
            continue
        
        try:
            # get the camera core-frame from front camera with dnat anonymization
            camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.DNAT)

            # get the image
            image = camera_core_frame.read()

            # get the ego road annotations
            polygon_annotations = zod_frame.get_annotation(AnnotationProject.EGO_ROAD)

            # convert the polygons to a binary mask (which can be used
            mask = polygons_to_binary_mask(polygon_annotations)
        except:
            continue

        seg_annotated_frames.append(frame_id)

    return seg_annotated_frames

def save_to_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))
        
def load_from_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_frame_seg_mask(zod_frames, frame_id):
    zod_frame = zod_frames[frame_id]

    # get the camera core-frame from front camera with dnat anonymization
    camera_core_frame = zod_frame.info.get_key_camera_frame(Anonymization.DNAT)

    # get the image
    image = camera_core_frame.read()

    # get the ego road annotations
    polygon_annotations = zod_frame.get_annotation(AnnotationProject.EGO_ROAD)

    # convert the polygons to a binary mask (which can be used
    mask = polygons_to_binary_mask(polygon_annotations)
    
    return mask

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

def main():
    zod_frames, training_frames, validation_frames = load_zod()

    train_frame_ids = get_seg_annotated_frame_ids(zod_frames, training_frames)
    val_frame_ids = get_seg_annotated_frame_ids(zod_frames, validation_frames)

    print(f'len train_frame_ids: ', len(train_frame_ids))
    print(f'len val_frame_ids: ', len(val_frame_ids))

    # save train/val annotated frameids
    save_to_json('training_seg_annotated_frames.json', train_frame_ids)
    save_to_json('validation_seg_annotated_frames.json', val_frame_ids)
