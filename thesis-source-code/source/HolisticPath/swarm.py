from utilities import *
from datasets import *


class SwarmClient:
    class Status(enum.Enum):
        READY = 1
        STARTED_GLOBAL_ROUND = 2
        BUZY = 3

    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        nr_local_epochs=NUM_LOCAL_EPOCHS,
        tb_path=None,
        swarm_subpath=None,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.losses = []
        self.val_losses = []
        self.neighbours = set([])
        self.recieved_from = set([])
        self.neighbours_agg_model = OrderedDict()
        self.status = SwarmClient.Status.READY
        self.absance_thresould = 0
        self.check_interval = 5  # seconds
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.swarm_subpath = swarm_subpath
        self.tb_writer = SummaryWriter(self.tb_path)

    def get_parameters(self):
        return self.net.state_dict()

    def fit(self, round_info):
        print(f"○ [Client {self.cid}] started local training - {round_info}")
        losses, accs, val_losses, val_accs = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs=self.nr_local_epochs,
            contin_val=True,
            plot=True,
            verbose=0,
            client_cid=self.cid,
            model_name=f"client {self.cid} - {round_info}",
            server_round=round_info,
            tb_writer=self.tb_writer,
            tb_subpath=f"{self.swarm_subpath}/{self.cid}/",
        )

        self.losses.append(losses)
        self.val_losses.append(val_losses)

    def participate_in_global_round(self, clients_network, edges, round_info):
        """spawn a thread that run the client participation logic
        then aggregates asyncronously"""
        print(f"🚧🚧🚧[Client {self.cid}] {round_info} has started.🚧🚧🚧")
        self.update_status(SwarmClient.Status.STARTED_GLOBAL_ROUND)
        t = threading.Thread(
            target=self.async_participate_in_global_round,
            args=(
                clients_network,
                edges,
                round_info,
            ),
        )
        t.start()

    def async_participate_in_global_round(self, clients_network, edges, round_info):
        try:
            self.discover_neighbours(edges)
            self.fit(round_info)
            self.broadcast_to_neighbours(clients_network)

            while not self.is_time_to_aggregate():
                print(
                    f"[Client {self.cid}] waiting until recieved sufficient amount of models..."
                )
                time.sleep(self.check_interval)
            print(
                f"[Client {self.cid}] got sufficient amount of models! started aggregating.."
            )
            self.update_status(SwarmClient.Status.BUZY)
            self.aggregate()
            self.update_status(SwarmClient.Status.READY)
            print(
                f"🚧🚧🚧[Client {self.cid}] {round_info} is done for me and I killed the local thread doing it.🚧🚧🚧"
            )
            return
        except BaseException as e:
            print(
                f"🔥🔥🔥[Client {self.cid}] Failed to continue participate_in_global_round: "
                + str(e)
            )
            return

    def is_time_to_aggregate(self):
        return len(self.recieved_from) >= (
            len(self.neighbours) - self.absance_thresould
        )

    def aggregate(self, strategy=None):
        print(f"-> [Client {self.cid}] validating before aggregating:")
        self.validate()
        print(
            f"[Client {self.cid}] aggredated parameters with {self.recieved_from} using {('FedAvg' if not strategy else strategy)} method"
        )
        if not strategy:
            self.FedAvg()
        print(f"<- [Client {self.cid}] validating After aggregating:")
        self.validate()

    def FedAvg(self):
        averaged_weights = self.get_parameters()

        for key in averaged_weights.keys():
            averaged_weights[key] = averaged_weights[key] + self.neighbours_agg_model[
                key
            ] / (len(self.recieved_from) + 1)

        self.net.load_state_dict(averaged_weights)
        del averaged_weights
        self.recieved_from = set([])
        self.neighbours_agg_model = OrderedDict()

    def validate(self):
        loss, accuracy = test(self.net, self.valloader)
        if ML_TASK == TASK.CLASSIFICATION:
            print(f"🌠 [Client {self.cid}] test loss {loss}, accuracy {accuracy}")
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        else:
            print(f"🌠 [Client {self.cid}] test RMSE {loss}")
            return float(loss), len(self.valloader), {"loss": float(loss)}

    def discover_neighbours(self, edges):
        print(f"🔭 [Client {self.cid}] discovering neighbours...")
        for edge in edges:
            if edge[1] == self.cid:
                self.neighbours.add(edge[0])
            elif edge[0] == self.cid:
                self.neighbours.add(edge[1])
        print(
            f"🔭 [Client {self.cid}] establised connection with the neighbours: {self.neighbours}"
        )

    def broadcast_to_neighbours(self, clients_network):
        print(
            f"📡 [Client {self.cid}] broadcasting the model to the neighbours: {self.neighbours}"
        )
        for n in self.neighbours:
            clients_network[n].recieve(self.cid, self.get_parameters())

    def recieve(self, sender_cid, params):
        if self.status == SwarmClient.Status.BUZY:
            print(
                f"X📡X [Client {self.cid}] descarded received model from: {sender_cid}. Too late!"
            )
            return

        if not sender_cid in self.recieved_from:
            self.recieved_from.add(sender_cid)
            self.add_params_to_buffer(params)
        print(f"📡 [Client {self.cid}] received a model from: {sender_cid}")

    def add_params_to_buffer(self, params, factor=1):
        if not self.neighbours_agg_model:
            self.neighbours_agg_model = params
        else:
            for key in params:
                self.neighbours_agg_model[key] += params[key] * factor

    def send_model(self, to):
        to.recieve(self.get_parameters())

    def update_status(self, status):
        print(f"[Client {self.cid}] updated status to {str(status)}")
        self.status = status

    def plot_typology_graph(self):
        G = nx.DiGraph()
        nodes = self.neighbours.copy()
        nodes.add(self.cid)
        G.add_nodes_from(nodes)
        G.add_edges_from([(self.cide, n) for n in nodes])
        nx.draw_networkx(
            G,
            bbox=dict(facecolor="skyblue", boxstyle="round", ec="silver", pad=1),
            edge_color="gray",
        )
        plt.title("Swarm learning simulation graph")
        plt.show()


class SwarmSimulator:
    def __init__(
        self,
        device,
        trainloaders,
        valloaders,
        testloader,
        nr_local_epochs=NUM_LOCAL_EPOCHS,
        tb_path=None,
        swarm_subpath=None,
    ):
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloader = testloader
        self.device = device
        self.nr_local_epochs = nr_local_epochs
        self.tb_path = tb_path
        self.swarm_subpath = swarm_subpath

    def create_client(self, cid):
        net = net_instance(f"client {cid}")
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        client = SwarmClient(
            cid,
            net,
            trainloader,
            valloader,
            nr_local_epochs=self.nr_local_epochs,
            tb_path=self.tb_path,
            swarm_subpath=self.swarm_subpath,
        )
        return client

    def perform_global_round(
        self, clients_network, edges, round_number, client_cid_subset=None
    ):
        print(
            f"""⦿⦿⦿⦿⦿  [SWARM SIMULATOR] started global round {round_number} between the clients 
              {client_cid_subset if client_cid_subset else [c.cid for c in clients_network]} ⦿⦿⦿⦿⦿ """
        )

        if client_cid_subset:
            for c_idx in client_cid_subset:
                client_edges = [e for e in edges if e[0] == c_idx or e[1] == c_idx]
                clients_network[c_idx].participate_in_global_round(
                    clients_network, client_edges, round_info=round_number
                )
        else:
            for c in clients_network:
                client_edges = [e for e in edges if e[0] == c.cid or e[1] == c.cid]
                clients_network[c.cid].participate_in_global_round(
                    clients_network, client_edges, round_info=round_number
                )

    def create_fully_connected_graph(self, clients_network):
        nodes = []
        edges = []
        for i in range(len(clients_network)):
            nodes.append(i)
            for j in range(i + 1, len(clients_network)):
                edges.append((i, j))
        return nodes, edges

    def wait_until_all_ready(self, clients_network, client_cid_subset=None):
        if client_cid_subset:
            while any(
                [
                    not clients_network[c].status == SwarmClient.Status.READY
                    for c in client_cid_subset
                ]
            ):
                time.sleep(2)
        else:
            while any(
                [not c.status == SwarmClient.Status.READY for c in clients_network]
            ):
                time.sleep(2)

    def simulate_fully_connected_graph(
        self, nr_clients=NUM_CLIENTS, nr_global_rounds=NUM_GLOBAL_ROUNDS
    ):
        # create and add clients to clients network
        clients_network = [self.create_client(i) for i in range(nr_clients)]

        nodes, edges = self.create_fully_connected_graph(clients_network)

        for i in range(nr_global_rounds):
            # perform one global rounds on all
            self.perform_global_round(clients_network, edges, round_number=i)

            # wait untill all clients updated their status to ready
            self.wait_until_all_ready(clients_network)

            print("⦿⦿⦿⦿⦿ [SWARM SIMULATOR] end global round ⦿⦿⦿⦿⦿ ")
        return nodes, edges

    def simulate_random_dynamic_graph(
        self, nr_clients=NUM_CLIENTS, nr_global_rounds=NUM_GLOBAL_ROUNDS
    ):
        # create and add clients to clients network
        clients_network = [self.create_client(i) for i in range(nr_clients)]

        nodes = []
        edges = []
        for i in range(nr_global_rounds):
            random_edges = [self.get_random_edge(clients_network) for _ in range(3)]
            nodes_of_edges = self.get_nodes_from_edges(random_edges)

            # perform one global rounds on all
            self.perform_global_round(
                clients_network, edges, client_cid_subset=nodes_of_edges, round_number=i
            )

            # wait untill all clients updated their status to ready
            self.wait_until_all_ready(clients_network, nodes_of_edges)

            print("⦿⦿⦿⦿⦿  [SWARM SIMULATOR] end global round ⦿⦿⦿⦿⦿ ")
            edges.extend(random_edges)
            nodes.extend(nodes_of_edges)

        return nodes, edges

    def get_nodes_from_edges(self, edges):
        return set([item for t in edges for item in t])

    def get_random_edge(self, clients_network):
        c_1_cid = random.choice(clients_network).cid
        c_2_cid = random.choice(clients_network).cid
        while c_1_cid == c_2_cid:
            c_2_cid = random.choice(clients_network).cid
        return (c_1_cid, c_2_cid)

    def plot_typology_graph(self, nodes, edges):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        nx.draw_networkx(
            G,
            bbox=dict(facecolor="skyblue", boxstyle="round", ec="silver", pad=1),
            edge_color="gray",
        )
        plt.title("⦿ [SWARM SIMULATOR] swarm learning simulation graph")
        plt.show()


def main(
    nr_clients=5,
    nr_local_epochs=10,
    nr_global_rounds=3,
    subset_factor=0.1,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    tb_path=TB_PATH,
    tb_swarm=TB_SWARM_SUB_PATH,
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
    swarm_sim = SwarmSimulator(
        device,
        trainloaders,
        valloaders,
        testloader,
        nr_local_epochs=nr_local_epochs,
        tb_path=tb_path,
        swarm_subpath=tb_swarm,
    )

    # simulate swarm learning with static fully connected graph typology
    swarm_sim.simulate_fully_connected_graph(
        nr_clients=nr_clients, nr_global_rounds=nr_global_rounds
    )

    # simulate swarm learning with dynamic random growing graph typology
    # swarm_sim.simulate_random_dynamic_graph(nr_clients=2, nr_global_rounds=nr_global_rounds)


if __name__ == "__main__":
    main()
