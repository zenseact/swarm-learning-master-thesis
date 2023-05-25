import unittest
from edge_code.data_loader import fleet_log
from server_code.data_partitioner import partition_train_data, PartitionStrategy
from edge_code.data_loader import load_datasets
from logging import INFO
class TestDataLoad(unittest.TestCase):

    def test_dataload(self):
        partitions = partition_train_data(PartitionStrategy.RANDOM, 10000)
        fleet_log(INFO,partitions['0'])
        fleet_log(INFO,partitions['1'])
        train, val, test = load_datasets(partitions['0'])
        fleet_log(INFO,train)
        fleet_log(INFO,type(train))

if __name__ == '__main__':
    unittest.main()