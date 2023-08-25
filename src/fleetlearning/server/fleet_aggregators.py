import os
from typing import Optional, List, Tuple
from flwr.common.typing import NDArrays
from logging import INFO
from flwr.server.strategy.aggregate import aggregate
from fleetlearning.common.logger import fleet_log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseStrategy:
    def __init__(self):
        pass

    def aggregate_fit_fedavg(
        self,
        results: List[Tuple[List, str]],
    ) -> Optional[NDArrays]:
        print("[SERVER] Aggregation started")
        res_formatted = [
            (weights, 1) for weights, _ in results
        ]  # the "1" is for weighting by e.g. number of training examples
        parameters_aggregated = aggregate(res_formatted)
        print("[SERVER] Aggregation done")
        return parameters_aggregated, {}
