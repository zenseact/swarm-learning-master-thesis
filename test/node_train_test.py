import unittest
from edge_code.data_loader import *
from server_code.data_partitioner import *
from common.static_params import *
from common.utilities import train

class TestNodeTrain(unittest.TestCase):

    def test_node_train(self):
        partitions = partition_train_data(PartitionStrategy.RANDOM, 10000)
        train, val, test = load_datasets(partitions['0'])
        train()

if __name__ == '__main__':
    unittest.main()