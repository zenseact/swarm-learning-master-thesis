import unittest
from edge_code.data_loader import *
from server_code.data_partitioner import *
from common.static_params import *
from common.utilities import train
from edge_com.edge_com import *
from edge_com.edge_handler import *

class TestNodeTrain(unittest.TestCase):

    def test_node_train(self):
        partition_train_data(PartitionStrategy.RANDOM, 1000)
        edge_com = EdgeCom(EdgeHandler(1))
        asd = edge_com.update_model("1")
        log(INFO,type(asd))

if __name__ == '__main__':
    unittest.main()