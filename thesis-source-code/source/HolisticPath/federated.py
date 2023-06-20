from utilities import *
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
        self.losses = []
        self.val_losses = []
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.federated_subpath = federated_subpath
        self.tb_writer = SummaryWriter(self.tb_path)

    def get_parameters(self, config):
        print(f"⤺ Get model parameters of client {self.cid}]")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"○ started local training of client {self.cid}]")
        set_parameters(self.net, parameters)
        losses, accs, val_losses, val_accs = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs=self.nr_local_epochs,
            contin_val=True,
            plot=True,
            verbose=0,
            client_cid=self.cid,
            model_name=f"client {self.cid}",
            server_round=config["server_round"],
            tb_writer=self.tb_writer,
            tb_subpath=f"{self.federated_subpath}/{self.cid}/",
        )
        self.losses.append(losses)
        self.val_losses.append(val_losses)
        params = get_parameters(self.net)
        save_model(self.net, self.cid)
        return params, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        if not self.net:
            self.net = load_model(self.cid)
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        if ML_TASK == TASK.CLASSIFICATION:
            print(f"🌠 [Client {self.cid}] test loss {loss}, accuracy {accuracy}")
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        else:
            print(f"🌠 [Client {self.cid}] test RMSE {loss}")
            return float(loss), len(self.valloader), {"loss": float(loss)}


class FederatedSimulator:
    def __init__(
        self,
        device,
        trainloaders,
        valloaders,
        testloader,
        nr_local_epochs=NUM_LOCAL_EPOCHS,
        tb_path=None,
        federated_subpath=None,
    ):
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloader = testloader
        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        self.client_resources = None
        if device.type == "cuda":
            self.client_resources = {"num_gpus": 0.2}
        self.device = device
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.federated_subpath = federated_subpath

    # The `evaluate` function will be by Flower called after every round
    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = net_instance(f"server")
        valloader = self.testloader
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, valloader)
        save_model(net, "server")

        writer = SummaryWriter(self.tb_path)

        writer.add_scalars(
            self.federated_subpath,
            {"global": np.mean(float(loss))},
            server_round,
        )

        writer.close()

        if ML_TASK == TASK.CLASSIFICATION:
            print(
                f"Server-side evaluation loss {float(loss)} / accuracy {float(accuracy)}"
            )
            return float(loss), {"accuracy": float(accuracy)}
        else:
            print(f"Server-side evaluation loss {float(loss)}")
            return float(loss), {}

    def on_fit_config_fn(self, server_round: int):
        return dict(server_round=server_round)

    def create_server_strategy(
        self,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    ):
        # Pass parameters to the Strategy for server-side parameter initialization
        server_model = net_instance(f"server")
        server_params = get_parameters(server_model)
        # destroy_model(server_model, "server")
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

    def sim_fed(self, nr_clients=NUM_CLIENTS, nr_global_rounds=NUM_GLOBAL_ROUNDS):
        # start federated learning simulation
        ram_memory = 10000 * 1024 * 1024

        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=nr_clients,
            config=fl.server.ServerConfig(num_rounds=nr_global_rounds),
            client_resources=self.client_resources,
            strategy=self.create_server_strategy(),
            ray_init_args={"object_store_memory": ram_memory},
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


def main(
    nr_clients=5,
    nr_local_epochs=10,
    nr_global_rounds=3,
    subset_factor=0.1,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    tb_path=TB_PATH,
    tb_federated=TB_FEDERATED_SUB_PATH,
):
    # import Zod data into memory
    zod = ZODImporter(
        subset_factor=subset_factor,
        img_size=img_size,
        batch_size=batch_size,
        tb_path=tb_path,
        stored_gt_path=STORED_GROUND_TRUTH_PATH,
    )

    # create pytorch loaders
    (
        trainloaders,
        valloaders,
        testloader,
        completeTrainloader,
        completeValloader,
    ) = zod.load_datasets(nr_clients)

    # create federated simulator
    fed_sim = FederatedSimulator(
        device,
        trainloaders,
        valloaders,
        completeValloader,
        nr_local_epochs=nr_local_epochs,
        tb_path=tb_path,
        federated_subpath=tb_federated,
    )

    # simulate federated learning
    fed_sim.sim_fed(nr_clients=nr_clients, nr_global_rounds=nr_global_rounds)


if __name__ == "__main__":
    main()
