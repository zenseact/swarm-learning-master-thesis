"""
Unit test of ZOD dataset loader and partitioner.
"""

from logging import INFO
import torch
import pytest

from server_code.data_partitioner import partition_train_data
from common.static_params import PartitionStrategy
from common.logger import fleet_log
from common.static_params import global_configs


@pytest.fixture(autouse=True)
def mock_static_params(mocker):
    mocked_global_config_dict = {
        "NUM_CLIENTS": 1000,
        "PERCENTAGE_OF_DATA": 0.02,
        "IMG_SIZE": 256,
        "BATCH_SIZE": 8,
        "VAL_FACTOR": 0.1,
        "SUBSET_FACTOR": 0.003,
        "OUTPUT_SIZE": 66,
        "NUM_OUTPUT": 51,
    }

    for key, value in mocked_global_config_dict.items():
        mocker.patch.object(global_configs, key, value)

    return mocked_global_config_dict


def test_dataloader(mock_static_params):

    # partition data among clients
    partitions = partition_train_data(
        PartitionStrategy.RANDOM, mock_static_params["NUM_CLIENTS"]
    )
    assert len(partitions.keys()) == mock_static_params["NUM_CLIENTS"]

    fleet_log(INFO, partitions['0'][:5])
    fleet_log(INFO, partitions['1'][:5])

    # create dataloaders for first client
    from edge_code.data_loader import load_datasets
    train_loader, val_loader, test_loader = load_datasets(partitions['0'])
    fleet_log(INFO, train_loader)
    fleet_log(INFO, type(train_loader))

    # assert the dataloaders are valid
    train_data, train_targets = next(iter(train_loader))
    val_data, val_targets = next(iter(val_loader))
    test_data, test_targets = next(iter(test_loader))

    data_shape = torch.Size(
        [
            mock_static_params["BATCH_SIZE"],
            3,
            mock_static_params["IMG_SIZE"],
            mock_static_params["IMG_SIZE"],
        ]
    )
    targets_shape = torch.Size(
        [mock_static_params["BATCH_SIZE"], mock_static_params["NUM_OUTPUT"]]
    )

    assert train_data.shape == data_shape
    assert train_targets.shape == targets_shape
    assert val_data.shape == data_shape
    assert val_targets.shape == targets_shape
    assert test_data.shape == data_shape
    assert test_targets.shape == targets_shape


def main():
    test_dataloader()


if __name__ == '__main__':
    main()
