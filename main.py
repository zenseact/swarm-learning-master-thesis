from common.static_params import *
from common.datasets import *
from server_code.federated_starter import FederatedStarter
import ray
from flwr.common.logger import log
from logging import INFO

def main(
        nr_clients=NUM_CLIENTS,
        nr_local_epochs=10,
        nr_global_rounds=3,
        subset_factor=0.1,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        tb_path=TB_PATH,
        tb_federated=TB_FEDERATED_SUB_PATH):
    # Initialize Ray
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    ray.init(**ray_init_args)  # type: ignore
    log(
        INFO,
        "Flower VCE: Ray initialized with resources: %s",
        ray.cluster_resources(),  # type: ignore
    )

    # import Zod data into memory
    zod = ZODImporter(subset_factor=subset_factor, img_size=img_size, batch_size=batch_size, tb_path=tb_path, stored_gt_path=STORED_GROUND_TRUTH_PATH)

    # create pytorch loaders, CHANGE TO ONLY LOAD TEST
    testloader = zod.load_datasets(nr_clients)

    # create federated simulator
    fed_sim = FederatedStarter(testloader, nr_local_epochs=nr_local_epochs, tb_path=tb_path, federated_subpath=tb_federated)

    # simulate federated learning
    fed_sim.sim_fed(nr_clients=nr_clients, nr_global_rounds=nr_global_rounds)


if __name__ == "__main__":
    main()