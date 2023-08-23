"""
Pipeline test of client side, with server side mocked.
"""
import os
import pathlib
import pytest
import logging
import numpy as np

from src.common.static_params import PartitionStrategy, global_configs
from src.common.utilities import net_instance, get_parameters
from src.server.data_partitioner import partition_train_data
from tests.utils.cleanup import cleanup_modules


@pytest.fixture(autouse=True)
def mock_static_params(mocker):
    mocked_global_config_dict = {
        "NUM_CLIENTS": 1000,  # in order to make the data partition small
        "PERCENTAGE_OF_DATA": 0.02,
        "IMG_SIZE": 256,
        "RUN_PRETRAINED": False,
        "BATCH_SIZE": 8,
        "VAL_FACTOR": 0.1,
        "SUBSET_FACTOR": 0.003,
        "NUM_LOCAL_EPOCHS": 1,
        "OUTPUT_SIZE": 66,
    }

    for key, value in mocked_global_config_dict.items():
        mocker.patch.object(global_configs, key, value)

    return mocked_global_config_dict


@pytest.fixture()
def partitions(mock_static_params):
    return partition_train_data(
        PartitionStrategy.RANDOM, mock_static_params["NUM_CLIENTS"]
    )


@pytest.fixture()
def agg():
    server_model = net_instance("server")
    return get_parameters(server_model)


@pytest.fixture()
def command_line_args():
    return [None, "0"]


@pytest.fixture(autouse=True)
def mock_server_comunication(mocker, agg, partitions):
    mocker.patch("edge_main.SSHClient")
    mocker.patch("edge_main.os.remove")
    mocker.patch("edge_main.list", return_value=agg)
    mocker.patch("edge_main.np.load", side_effect=[{"arr_0": "dummy_agg"}, partitions])


def test_pipeline_client(caplog, mocker, command_line_args):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    for file in os.listdir(tmp_dir):
        if file == "res0.npz":
            os.remove(os.path.join(ROOT, "tmp", file))

    # mock GPU usage
    mocker.patch("edge_main.use_gpu")

    # run main script
    import src.scheduler.train as train

    mocker.patch.object(train.sys, "argv", command_line_args)
    train.main()
    mocker.stopall()

    # assert that correct np file is created and can be loaded
    assert "res0.npz" in os.listdir(tmp_dir)
    np.load(os.path.join(tmp_dir, "res0.npz"), allow_pickle=True)["arr_0"]

    assert "done" in caplog.text


def main():
    test_pipeline_client()


if __name__ == "__main__":
    main()

cleanup_modules()
