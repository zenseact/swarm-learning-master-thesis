from common.utilities import *
from common.datasets import *
from server_code.clients.flwr_client import FlowerClient
from edge_com.edge_handler import EdgeHandler
from server_code.strategies.base_strategy import BaseStrategy
import flwr as fl
from flwr.common.typing import Optional, Tuple, Dict

class FederatedStarter:
    def __init__(self, testloader, nr_local_epochs=NUM_LOCAL_EPOCHS, tb_path=None, federated_subpath=None):
        self.edge_handler = EdgeHandler(1)
        self.testloader = testloader
        self.client_resources = None
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.federated_subpath = federated_subpath

    # The `evaluate` function will be by Flower called after every round
    def evaluate(self,
                 server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
                 ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = net_instance(f"server")
        valloader = self.testloader
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, valloader)
        save_model(net, "server")

        # writer = SummaryWriter(self.tb_path)

        # writer.add_scalars(
        #     self.federated_subpath,
        #     {"global": np.mean(float(loss))},
        #     server_round,
        # )

        # writer.close()

        if (ML_TASK == TASK.CLASSIFICATION):
            print(f"Server-side evaluation loss {float(loss)} / accuracy {float(accuracy)}")
            return float(loss), {"accuracy": float(accuracy)}
        else:
            print(f"Server-side evaluation loss {float(loss)}")
            return float(loss), {}

    def on_fit_config_fn(self, server_round: int):
        return dict(server_round=server_round)

    def create_server_strategy(self,
                               fraction_fit=1, fraction_evaluate=1, min_fit_clients=NUM_CLIENTS,
                               min_evaluate_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS):
        # Pass parameters to the Strategy for server-side parameter initialization
        server_model = net_instance(f"server")
        server_params = get_parameters(server_model)
        print('Saving initial parameters for edge devices')
        np.savez(get_parameters(server_params), "../tmp/agg.npz")
        strategy = BaseStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(server_params),
            evaluate_fn=self.evaluate,
            on_fit_config_fn=self.on_fit_config_fn
        )
        return strategy

    def sim_fed(self, nr_clients=NUM_CLIENTS, nr_global_rounds=NUM_GLOBAL_ROUNDS):
        # start federated learning simulation
        
        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=nr_clients,
            config=fl.server.ServerConfig(num_rounds=nr_global_rounds),
            client_resources=self.client_resources,
            strategy=self.create_server_strategy(),
        )
        

    def client_fn(self, cid) -> FlowerClient:
        client = FlowerClient(cid, self.edge_handler)
        return client
