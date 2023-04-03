import unittest
from edge_code.data_loader import *
from server_code.data_partitioner import *
from common.static_params import *

class TestAddNumbers(unittest.TestCase):

    def test_dataload(self):
        partitions = partition_train_data(PartitionStrategy.RANDOM, 10000)
        print(partitions['0'])
        print(partitions['1'])

if __name__ == '__main__':
    unittest.main()