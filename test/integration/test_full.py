"""
Pipeline test off complete FL session with real AGX nodes and no mocking.

Please note that all changes need to be pushed to git in order to be applicable for
clients.
"""

import os
import pathlib
import logging
import numpy as np

import main as server_main
from common.static_params import global_configs
from test.utils.cleanup import cleanup_modules


def test_pipeline(
    caplog,
):
    logging.getLogger(__name__)

    # delete old files if present
    ROOT = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
    tmp_dir = os.path.join(ROOT, "tmp")
    for file in os.listdir(tmp_dir):
        if file.split(".")[-1] == "npz" and file.split(".")[0] != "res0":
            os.remove(os.path.join(ROOT, "tmp", file))

    # run main script
    server_main.main()

    # assert results are retrieved by server
    assert "agg.npz" in os.listdir(tmp_dir)
    assert "partitions.npz" in os.listdir(tmp_dir)

    np.load(os.path.join(tmp_dir, "partitions.npz"))["0"]
    np.load(os.path.join(tmp_dir, "agg.npz"), allow_pickle=True)['arr_0']

    # assert info log content
    n_clients = global_configs.NUM_CLIENTS

    log_contents = [
        "Ray initialized",
        f"fit_round 1: strategy sampled {n_clients} clients (out of {n_clients})",
        f"fit_round 1 received {n_clients} results and 0 failures",
        f"evaluate_round 1: strategy sampled {n_clients} clients (out of {n_clients})",
        f"evaluate_round 1 received 0 results and {n_clients} failures",
        "FL finished",
    ]

    for log_content in log_contents:
        assert log_content in caplog.text


def main():
    test_pipeline()


if __name__ == "__main__":
    main()

cleanup_modules()
