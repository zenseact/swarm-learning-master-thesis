from common.static_params import global_configs
from common.datasets import ZODImporter
from server_code.federated_starter import FederatedStarter
import ray
from common.logger import fleet_log
from logging import INFO

def main(
        nr_clients=global_configs.NUM_CLIENTS,
        nr_local_epochs=global_configs.NUM_LOCAL_EPOCHS,
        nr_global_rounds=global_configs.NUM_GLOBAL_ROUNDS,
        subset_factor=global_configs.SUBSET_FACTOR,
        img_size=global_configs.IMG_SIZE,
        batch_size=global_configs.BATCH_SIZE,
        device=global_configs.DEVICE,
        tb_path=global_configs.TB_PATH,
        tb_federated=global_configs.TB_FEDERATED_SUB_PATH):
    # Initialize Ray
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "object_store_memory": global_configs.GB_RAM * 1024 * 1024 * 1024/(global_configs.NUM_CLIENTS), # for some reason clients have to share 1 GB memory.. But its fine probably
        "num_cpus" : global_configs.NUM_CPUS,
    }
    ray.init(**ray_init_args)  # type: ignore
    fleet_log(
        INFO,
        "Flower VCE: Ray initialized with resources: %s",
        ray.cluster_resources(),  # type: ignore
    )

    # import Zod data into memory
    zod = ZODImporter(subset_factor=subset_factor, img_size=img_size, batch_size=batch_size, tb_path=tb_path, stored_gt_path=global_configs.STORED_GROUND_TRUTH_PATH)

    # create pytorch loaders, CHANGE TO ONLY LOAD TEST
    testloader = zod.load_datasets(nr_clients)
    fleet_log(INFO,f"len testloader batches: {len(testloader)}")

    # create federated simulator
    fed_sim = FederatedStarter(testloader, nr_local_epochs=nr_local_epochs, tb_path=tb_path, federated_subpath=tb_federated)

    # simulate federated learning
    fed_sim.sim_fed(nr_clients=nr_clients, nr_global_rounds=nr_global_rounds)


if __name__ == "__main__":
    main()