from common.static_params import PartitionStrategy
from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
import random

# partitioning is done from server side in data_partitioner.py
def load_train_data(partitioned_frame_ids: list):
    pass

def load_test_data():
    pass