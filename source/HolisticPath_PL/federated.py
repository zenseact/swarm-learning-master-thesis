from train import *
from datasets import *


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        nr_local_epochs,
        tb_path=None,
        federated_subpath=None,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.federated_subpath = federated_subpath
        self.tb_writer = SummaryWriter(self.tb_path)

    def fit(self, parameters, config):
        print(f"○ started local training of client {self.cid}]")
        set_parameters(self.net, parameters, self.cid)
        train(
            self.net,
            self.trainloader,
            self.valloader,
            nr_epochs=self.nr_local_epochs,
        )
        params = get_parameters(self.net, self.cid)
        save_model(self.net, self.cid)
        return params, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        if not self.net:
            self.net = load_model(self.cid)
        set_parameters(self.net, parameters, self.cid)
        loss = test(self.net, self.valloader)

        print(f"🌠 [Client {self.cid}] test RMSE {loss}")
        return float(loss), len(self.valloader), {"loss": float(loss)}


class FederatedSimulator:
    def __init__(
        self,
        device,
        trainloaders,
        valloaders,
        testloader,
        nr_local_epochs=c('num_local_epochs'),
        tb_path=TB_PATH,
        federated_subpath=TB_FEDERATED_SUB_PATH,
    ):
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloader = testloader
        self.client_resources = {"num_gpus": 1}
        self.device = device
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.federated_subpath = federated_subpath
        self.name = 'SERVER'

    # The `evaluate` function will be by Flower called after every round
    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = net_instance(self.name)
        testloader = self.testloader
        set_parameters(net, parameters, cid=self.name)  # Update model with the latest parameters
        loss = test(net, testloader)
        loss = loss[0]['test_loss']
        save_model(net, self.name)

        writer = SummaryWriter(self.tb_path)

        writer.add_scalars(
            self.federated_subpath,
            {"global": float()},
            server_round,
        )

        writer.close()
        print(f"Server evaluation loss {float(loss)}")
        return float(loss), {}

    def on_fit_config_fn(self, server_round: int):
        return dict(server_round=server_round)

    def create_server_strategy(
        self,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=c('num_clients'),
        min_evaluate_clients=c('num_clients'),
        min_available_clients=c('num_clients'),
    ):
        # Pass parameters to the Strategy for server-side parameter initialization
        server_model = net_instance(self.name)
        server_params = get_parameters(server_model, self.name)
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(server_params),
            evaluate_fn=self.evaluate,
            on_fit_config_fn=self.on_fit_config_fn,
        )
        return strategy

    def sim_fed(self, nr_clients=c('num_clients'), nr_global_rounds=c('num_global_rounds')):
        ram_memory = 10000 * 1024 * 1024

        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=nr_clients,
            config=fl.server.ServerConfig(num_rounds=nr_global_rounds),
            client_resources=self.client_resources,
            strategy=self.create_server_strategy(),
            #ray_init_args={"object_store_memory": ram_memory},
            ray_init_args={"num_gpus":1}
        )

    def client_fn(self, cid) -> FlowerClient:
        net = net_instance(f"client {cid}")
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        client = FlowerClient(
            cid,
            net,
            trainloader,
            valloader,
            self.nr_local_epochs,
            self.tb_path,
            self.federated_subpath,
        )
        return client


def main():

    # import Zod data into memory
    zod = ZODImporter()

    # create pytorch loaders
    (
        trainloaders,
        valloaders,
        testloader,
        completeTrainloader,
        completeValloader,
    ) = zod.load_datasets(c('num_clients'))

    # create federated simulator
    fed_sim = FederatedSimulator(
        DEVICE,
        trainloaders,
        valloaders,
        completeValloader,
        nr_local_epochs=c('num_local_epochs'),
    )

    # simulate federated learning
    fed_sim.sim_fed(nr_clients=c('num_clients'), nr_global_rounds=c('num_global_rounds'))


if __name__ == "__main__":
    main()
