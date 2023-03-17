from utilities import *
from datasets import *

app = None

class SwarmClient:
    class Status(enum.Enum):
        READY = 1
        STARTED_GLOBAL_ROUND = 2
        BUZY = 3

    def __init__(self, cid, net, trainloader, valloader, swarm_config, nr_local_epochs=NUM_LOCAL_EPOCHS):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.losses = []
        self.val_losses = []
        self.neighbours = set([])
        self.ready_neighbours = set([])
        self.recieved_from = set([])
        self.neighbours_agg_model = OrderedDict()
        self.status = SwarmClient.Status.READY
        self.absance_thresould = 0
        self.check_interval = 5  # seconds
        self.nr_local_epochs = nr_local_epochs
        self.swarm_config = swarm_config
        self.global_round_counter = 1

        # get own ip adress and port from the config
        client_ip, client_port = SwarmClient.get_client_adress(swarm_config, cid)
        self.adress = f"http://{client_ip}:{client_port}"

        # run flask server
        app = Flask(__name__)
        app.run(host=client_ip, port=client_port, debug=False)

        print(f"I'm the swarm client {cid} and I'm awake!")


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
            model_name=f"client {self.cid} - {round_info}"
        )

        self.losses.append(losses)
        self.val_losses.append(val_losses)

    def participate_in_global_round(self):
        try:
            """run the client participation logic then aggregates"""
            round_info = f'round {self.global_round_counter}'
            print(f"🚧🚧🚧[Client {self.cid}] {round_info} has started.🚧🚧🚧")
            self.update_status(SwarmClient.Status.STARTED_GLOBAL_ROUND)

            self.fit(round_info)
            self.broadcast_to_neighbours()

            while (not self.is_time_to_aggregate()):
                print(f"[Client {self.cid}] waiting until recieved sufficient amount of models...")
                time.sleep(self.check_interval)

            print(f"[Client {self.cid}] got sufficient amount of models! started aggregating..")
            self.update_status(SwarmClient.Status.BUZY)
            self.aggregate()
            self.update_status(SwarmClient.Status.READY)
            print(f"🚧🚧🚧[Client {self.cid}] {round_info} is done for me and I killed the local thread doing it.🚧🚧🚧")
            self.global_round_counter += self.global_round_counter
            return
        except BaseException as e:
            print(f"🔥🔥🔥[Client {self.cid}] Failed to continue participate_in_global_round: " + str(e))
            return

    def is_time_to_aggregate(self):
        return len(self.recieved_from) >= (len(self.neighbours) - self.absance_thresould)

    def aggregate(self, strategy=None):
        print(f"-> [Client {self.cid}] validating before aggregating:")
        self.validate()
        print(
            f"[Client {self.cid}] aggredated parameters with {self.recieved_from} using {('FedAvg' if not strategy else strategy)} method")
        if (not strategy):
            self.FedAvg()
        print(f"<- [Client {self.cid}] validating After aggregating:")
        self.validate()

    def FedAvg_(self):
        averaged_weights = OrderedDict()
        self_params = self.get_parameters()

        for key in self_params.keys():
            averaged_weights[key] = (self_params[key] + self.neighbours_agg_model[key]) / (len(self.recieved_from) + 1)

        self.net.load_state_dict(averaged_weights)
        del averaged_weights
        self.recieved_from = set([])
        self.neighbours_agg_model = OrderedDict()

    def FedAvg(self):
        averaged_weights = self.get_parameters()

        for key in averaged_weights.keys():
            averaged_weights[key] = (
                    averaged_weights[key] + self.neighbours_agg_model[key] / (len(self.recieved_from) + 1))

        self.net.load_state_dict(averaged_weights)
        del averaged_weights
        self.recieved_from = set([])
        self.neighbours_agg_model = OrderedDict()

    def validate(self):
        loss, accuracy = test(self.net, self.valloader)
        if (ML_TASK == TASK.CLASSIFICATION):
            print(f"🌠 [Client {self.cid}] test loss {loss}, accuracy {accuracy}")
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        else:
            print(f"🌠 [Client {self.cid}] test RMSE {loss}")
            return float(loss), len(self.valloader), {"loss": float(loss)}

    def discover_neighbours(self):
        print(f"🔭 [Client {self.cid}] discovering neighbours...")
        self.spawn_thread(method=self.send_heart_beats_async, args=())

        while(True):
            time.sleep(30)
            run_global_round = SwarmClient.get_run_global_round(self.swarm_config)
            if(run_global_round and self.all_neighbors_ready()):
                self.participate_in_global_round()

    
    def all_neighbors_ready(self):
        neighbours_status=[]
        for cid in self.neighbours:
            neighbours_status.append(SwarmClient.get_status(cid) == SwarmClient.Status.READY)
        return all(neighbours_status)

    def broadcast_to_neighbours(self):
        print(f"📡 [Client {self.cid}] broadcasting the model to the neighbours: {self.neighbours}")
        for cid in self.neighbours:
            SwarmClient.send_data(cid, 'recieve_model', params='', data={'sender_id':self.cid, 'params':self.get_parameters()})

    def recieve(self, sender_cid, params):
        if (self.status == SwarmClient.Status.BUZY):
            print(f"X📡X [Client {self.cid}] descarded received model from: {sender_cid}. Too late!")
            return

        if (not sender_cid in self.recieved_from):
            self.recieved_from.add(sender_cid)
            self.add_params_to_buffer(params)
        print(f"📡 [Client {self.cid}] received a model from: {sender_cid}")

    def add_params_to_buffer(self, params, factor=1):
        if (not self.neighbours_agg_model):
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
        nx.draw_networkx(G, bbox=dict(facecolor="skyblue",
                                      boxstyle="round", ec="silver", pad=1),
                                      edge_color="gray")
        plt.title("Swarm learning simulation graph")
        plt.show()

    def get_cid_list(self):
        cid_list = SwarmClient.get_client_ids(self.swarm_config)
        typology = SwarmClient.get_client_typology(self.swarm_config)

        if(typology == 'fully_connected'):
            return cid_list
        if(typology == 'random_dynamic'):
            nr_picked_neighbours = random.randint(2, len(self.neighbours))
            return random.sample(self.neighbours,nr_picked_neighbours)

    def send_heart_beats_async(self):
        while(True):
            neighbours = set([])
            for cid in self.get_cid_list():
                c_ip, c_port = SwarmClient.get_client_adress(cid)
                params = f'?cid={self.cid}&status={self.status}'
                status_code = SwarmClient.send_data(cid, "heart_beat", params, {}, f"sent heart beat to client {cid} with adress {c_ip}:{c_port}")
                if(status_code):
                    neighbours.add(cid)

            if(neighbours != self.neighbours and self.status == SwarmClient.Status.READY):
                self.neighbours = neighbours
                print(f"🔭 [Client {self.cid}] Detected new typology. Establised connection with the neighbours: {self.neighbours}")

            time.sleep(2)

    def spawn_thread(method, args):
        threading.Thread(target=method, args=args).start()


    """API endpoints"""
    @app.route("/heart_beat", methods = ['POST', 'GET'])
    def get_heart_beat(cid, status):
        print(f'got heart beat signal from {cid} with status {status}')

    @app.route("/status", methods = ['POST', 'GET'])
    def status(self):
        return self.status

    @app.route("/recieve_model", methods = ['POST'])
    def recieve_model(self):
        content = request.json
        sender_id = str(content['sender_id'])
        params = OrderedDict(content['params'])
        self.recieve(sender_id, params) 

    """static methods"""
    @classmethod
    def send_data(cid, route, params, data, log_message=None):
        try:
            c_ip, c_port = SwarmClient.get_client_adress(cid)
            r = requests.post(f"http://{c_ip}:{c_port}/{route}{params}", data)
            if(log_message):
                print(log_message) 
            return r.status_code == 200
        except:
            print(f'could not establish connection with {c_ip}:{c_port}')
            return None
        
    @classmethod
    def get_status(cid):
        try:
            c_ip, c_port = SwarmClient.get_client_adress(cid)
            status = requests.get(f"http://{c_ip}:{c_port}/status").content.decode('utf-8')
            print('status of {cid} is {status}')
            return status
        except:
            print(f'could not get status of {c_ip}:{c_port}')
            return None

    @classmethod
    def get_client_ids(swarm_config):
        client_ids_key = 'client_ids'
        return json.loads(swarm_config[client_ids_key])

    @classmethod
    def get_client_adress(swarm_config, cid):
        client_adresses_key = 'client_adresses'
        for c in swarm_config[client_adresses_key]:
            if list(c.keys())[0] == str(cid):
                c_address = (c[str(cid)]).split(':')
                return c_address[0], c_address[1]
        return None

    @classmethod
    def get_client_typology(swarm_config):
        client_typology_key = 'client_typology'
        return swarm_config[client_typology_key]

    @classmethod
    def get_nr_local_epochs(swarm_config):
        nr_local_epochs = swarm_config['nr_local_epochs']
        if(nr_local_epochs != None):
            return int(nr_local_epochs)
    
    @classmethod
    def get_subset_factor(swarm_config):
        subset_factor = swarm_config['subset_factor']
        if(subset_factor != None):
            return float(subset_factor)
    
    @classmethod
    def get_batch_size(swarm_config):
        batch_size = swarm_config['batch_size']
        if(batch_size != None):
            return int(batch_size)
    
    @classmethod
    def get_run_global_round(swarm_config):
        run_global_round = swarm_config['run_global_round']
        if(run_global_round != None):
            return bool(run_global_round)


def create_client(cid, trainloaders, valloaders, swarm_config, nr_local_epochs):
        net = net_instance(f"client {cid}")
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        client = SwarmClient(cid, net, trainloader, valloader, swarm_config, nr_local_epochs=nr_local_epochs)
        return client




def main(cid, url=None):

    # read the swarm config file
    url = 'https://ycommonstorage.blob.core.windows.net/misc/swarm_config.json?sp=r&st=2023-02-26T01:42:42Z&se=2026-01-01T09:42:42Z&sv=2021-06-08&sr=b&sig=dF8sGV7e%2FKmREMY%2FGWsYPhmuWyn8ysl%2BxVd5LvlY9Yc%3D' if url == None else url
    swarm_config = json.loads(requests.get(url).content.decode('utf-8'))
    print('Downloaded the swarm config file:\n',swarm_config)

    # load data from swarm config 
    nr_local_epochs = SwarmClient.get_nr_local_epochs(swarm_config)
    subset_factor = SwarmClient.get_subset_factor(swarm_config)
    batch_size = SwarmClient.get_batch_size(swarm_config)
    nr_clients = len(SwarmClient.get_client_ids(swarm_config))

    # import Zod data into memory
    zod = ZODImporter(subset_factor=subset_factor, img_size=IMG_SIZE, batch_size=batch_size)

    # create pytorch loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zod.load_datasets(nr_clients)

    # create swarm client
    c = create_client(cid, trainloaders, valloaders, swarm_config, nr_local_epochs)


if __name__ == "__main__":
    main()




