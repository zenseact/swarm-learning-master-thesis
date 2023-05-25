import unittest
from server_code.data_partitioner import partition_train_data, PartitionStrategy
from common.logger import fleet_log
from logging import INFO
from edge_com.edge_com import EdgeCom
from edge_com.edge_handler import EdgeHandler

class TestNodeTrain(unittest.TestCase):

    def test_node_train(self):
        partition_train_data(PartitionStrategy.RANDOM, 1000)
        edge_com = EdgeCom(EdgeHandler(1))
        asd = edge_com.update_model("1")
        fleet_log(INFO,type(asd))

if __name__ == '__main__':
    unittest.main()