

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import flwr as fl
from typing import List, Optional, Tuple, Union
from flwr.common.logger import log
from logging import WARNING
from flwr.server.strategy import FedAvg
import numpy as np
from common.utilities import get_parameters

from flwr.common import (
    FitRes,
)

from flwr.server.client_proxy import ClientProxy


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
        print('Saving newest parameters')
        np.savez(get_parameters(parameters_aggregated), "../../tmp/agg.npz")
        return parameters_aggregated, metrics_aggregated