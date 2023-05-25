"""
Pipeline test.
"""

import os
import pathlib
import pytest
import logging
import numpy as np
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, EvaluateRes
import ray

from common.utilities import net_instance, get_parameters
import main as server_main


@pytest.fixture()
def model_parameters():
    model = net_instance("1")
    params = get_parameters(model)
    params = list(np.array(params, dtype=object))
    return params


@pytest.fixture()
def fit_res(model_parameters):
    return FitRes(
        status=Status(code=Code(0), message='Success'),
        parameters=ndarrays_to_parameters(model_parameters),
        num_examples=1,
        metrics={},
    )


@pytest.fixture()
def eval_res():
    return EvaluateRes(
        status=Status(code=Code(0), message='Success'),
        loss=0.5,
        num_examples=1,
        metrics={},
    )


@pytest.fixture()
def ray_args():
    n_clients = 1
    return {
        "local_mode": True,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "object_store_memory": 1024 * 1024 * 1024/(n_clients*1.2),
        "num_cpus": 4,
    }


@pytest.fixture()
def static_params():
    return {
        "DEVICE_DICT": {"dummy_device_1": 0},
        "NUM_CLIENTS": 1,
        "PERCENTAGE_OF_DATA": 0.001,
        "IMG_SIZE": 256,
        "RUN_PRETRAINED": False,
        "BATCH_SIZE": 8,
        "VAL_FACTOR": 0.1,
        "SUBSET_FACTOR": 0.003,
        "NUM_GLOBAL_ROUNDS": 1,
        "NUM_LOCAL_EPOCHS": 1,
        "OUTPUT_SIZE": 66,
    }


def test_pipeline_agx(
    caplog,
    mocker,
    model_parameters,
    fit_res,
    eval_res,
    ray_args,
    static_params
):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    for file in os.listdir(tmp_dir):
        if file.split(".")[-1] == "npz" and file.split(".")[0] != "res0":
            os.remove(os.path.join(ROOT, "tmp", file))

    # change static params
    for key, value in static_params.items():
        mocker.patch(f"common.static_params.{key}", return_value=value)

    # turn on ray local mode
    mocker.patch("main.ray.init", return_value=ray.init(**ray_args))

    # mock all client communication
    mocker.patch("server_code.clients.flwr_client.EdgeCom.__init__", return_value=None)
    mocker.patch(
        "server_code.clients.flwr_client.EdgeCom.update_model",
        return_value=model_parameters
    )

    # mock ray returns of results
    mocker.patch("server_code.ray_client_proxy_flwr.ray.get", side_effect=[
            fit_res, eval_res
        ]
    )

    # run main script
    server_main.main()

    # assert results are retrieved by server
    assert "agg.npz" in os.listdir(tmp_dir)
    assert "partitions.npz" in os.listdir(tmp_dir)

    # assert info log contains
    assert "Ray initialized" in caplog.text
    assert "fit_round 1: strategy sampled 1 clients (out of 1)" in caplog.text
    assert "fit_round 1 received 1 results and 0 failures" in caplog.text
    assert "evaluate_round 1: strategy sampled 1 clients (out of 1)" in caplog.text
    assert "evaluate_round 1 received 1 results and 0 failures" in caplog.text
    assert "FL finished" in caplog.text


def main():
    test_pipeline_agx()


if __name__ == "__main__":
    main()