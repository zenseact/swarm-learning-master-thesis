"""
Pipeline test of server side, with client side mocked.
"""
import os
import pathlib
from importlib import reload
import pytest
import logging
import numpy as np
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, EvaluateRes
import ray

from src.common.static_params import global_configs


@pytest.fixture(autouse=True)
def mock_static_params(mocker):
    mocked_global_config_dict = {
        "DEVICE_DICT": {"dummy_device_1": 0},
        "NUM_CLIENTS": 1,  # currently only works for 1 client due to local ray instance
        "PERCENTAGE_OF_DATA": 0.02,
        "IMG_SIZE": 256,
        "RUN_PRETRAINED": False,
        "BATCH_SIZE": 8,
        "VAL_FACTOR": 0.1,
        "SUBSET_FACTOR": 0.003,
        "NUM_GLOBAL_ROUNDS": 1,
        "NUM_LOCAL_EPOCHS": 1,
        "OUTPUT_SIZE": 66,
    }

    for key, value in mocked_global_config_dict.items():
        mocker.patch.object(global_configs, key, value)

    return mocked_global_config_dict


@pytest.fixture()
def model_parameters():
    from src.common.utilities import net_instance, get_parameters

    model = net_instance("dummy_name")
    params = get_parameters(model)
    params = list(np.array(params, dtype=object))
    return params


@pytest.fixture()
def fit_res(model_parameters):
    return FitRes(
        status=Status(code=Code(0), message="Success"),
        parameters=ndarrays_to_parameters(model_parameters),
        num_examples=1,
        metrics={},
    )


@pytest.fixture()
def eval_res():
    return EvaluateRes(
        status=Status(code=Code(0), message="Success"),
        loss=0.5,
        num_examples=1,
        metrics={},
    )


@pytest.fixture(autouse=True)
def mock_ray_communication(mocker, mock_static_params, fit_res, eval_res):
    mocked_ray_args = {
        "local_mode": True,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "object_store_memory": 1024
        * 1024
        * 1024
        / (mock_static_params["NUM_CLIENTS"] * 1.2),
        "num_cpus": 4,
    }
    mocker.patch("main.ray.init", return_value=ray.init(**mocked_ray_args))
    mocker.patch(
        "server_code.ray_client_proxy_flwr.ray.get", side_effect=[fit_res, eval_res]
    )


@pytest.fixture(autouse=True)
def mock_client_communication(mocker, model_parameters):
    mocker.patch("server_code.clients.flwr_client.EdgeCom.__init__", return_value=None)
    mocker.patch(
        "server_code.clients.flwr_client.EdgeCom.update_model",
        return_value=model_parameters,
    )


def test_pipeline_server(caplog):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    for file in os.listdir(tmp_dir):
        if file.split(".")[-1] == "npz" and file.split(".")[0] != "res0":
            os.remove(os.path.join(ROOT, "tmp", file))

    # run main script
    import src.server.server_main as server_main

    reload(server_main)  # to avoid using cached non-mocked global params
    server_main.main()

    # assert results are retrieved by server
    assert "agg.npz" in os.listdir(tmp_dir)
    assert "partitions.npz" in os.listdir(tmp_dir)

    # assert saved files are valid numpys
    np.load(os.path.join(ROOT, "tmp", "partitions.npz"))["0"]
    np.load(os.path.join(ROOT, "tmp", "agg.npz"), allow_pickle=True)["arr_0"]

    # assert info log conent
    assert "Ray initialized" in caplog.text
    assert "fit_round 1: strategy sampled 1 clients (out of 1)" in caplog.text
    assert "fit_round 1 received 1 results and 0 failures" in caplog.text
    assert "evaluate_round 1: strategy sampled 1 clients (out of 1)" in caplog.text
    assert "evaluate_round 1 received 1 results and 0 failures" in caplog.text
    assert "FL finished" in caplog.text


def main():
    test_pipeline_server()


if __name__ == "__main__":
    main()
