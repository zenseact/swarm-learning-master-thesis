

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import flwr as fl
from typing import List, Optional, Tuple, Union
from flwr.common.logger import log
from logging import WARNING
from functools import reduce

from flwr.common import (
    FitRes,
)

from flwr.server.client_proxy import ClientProxy


class BaseStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[fl.common.NDArrays]:
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        return aggregated_weights