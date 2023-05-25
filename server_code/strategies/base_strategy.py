

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import flwr as fl
from typing import List, Optional, Tuple, Union
from common.logger import fleet_log
from logging import WARNING
from flwr.server.strategy import FedAvg
import numpy as np
from common.utilities import get_parameters
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy
from logging import INFO


class BaseStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[fl.common.NDArrays]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        fleet_log(INFO,'Saving newest parameters')
        np.savez("tmp/agg.npz", parameters_to_ndarrays(parameters_aggregated))
        return parameters_aggregated, metrics_aggregated