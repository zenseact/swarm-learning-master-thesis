from static_params import *
from datasets import main as run_datasets
from centralized import main as run_centralized
from federated import main as run_federated
from swarm import main as run_swarm
import argparse


def main():
    # ...
    parser = argparse.ArgumentParser(
        prog="holisticpath",
        usage="holistic path project runner",
        description="""choose one of the following modules to run and choose its
                        arguments. If Arguments are not specified, the default
                        arguments specified in static_parameters.py will apply. 
                        Note that centralized, federated and swarm choices doesn't neet you
                        to run datasets first

    modules:
    --module = dataset, centralized, federated and swarm

    args:
    --nr_clients
    --nr_local_epochs
    --nr_global_rounds
    --subset_factor
    --img_size
    --batch_size
    --device"""
    )
    parser.add_argument(
        "--module",
        type=str,
        required=False
    )
    parser.add_argument(
        "--nr_clients",
        default=NUM_CLIENTS,
        type=int,
        required=False
    )
    parser.add_argument(
        "--nr_local_epochs",
        default=NUM_LOCAL_EPOCHS,
        type=int,
        required=False
    )
    parser.add_argument(
        "--nr_global_rounds",
        default=NUM_GLOBAL_ROUNDS,
        type=int,
        required=False
    )
    parser.add_argument(
        "--subset_factor",
        default=SUBSET_FACTOR,
        type=float,
        required=False
    )
    parser.add_argument(
        "--val_factor",
        default=VAL_FACTOR,
        type=float,
        required=False
    )
    parser.add_argument(
        "--img_size",
        default=IMG_SIZE,
        type=int,
        required=False
    )
    parser.add_argument(
        "--batch_size",
        default=BATCH_SIZE,
        type=int,
        required=False
    )

    args = parser.parse_args()
    print(args)

    if (args.module == 'dataset'):
        run_datasets(nr_clients=args.nr_clients,
                     subset_factor=args.subset_factor,
                     img_size=args.img_size,
                     batch_size=args.batch_size)

    elif (args.module == 'centralized'):
        run_centralized(nr_clients=args.nr_clients,
                        nr_local_epochs=args.nr_local_epochs,
                        subset_factor=args.subset_factor,
                        img_size=args.img_size,
                        batch_size=args.batch_size)

    elif (args.module == 'federated'):
        run_federated(nr_clients=args.nr_clients,
                      nr_local_epochs=args.nr_local_epochs,
                      nr_global_rounds=args.nr_global_rounds,
                      subset_factor=args.subset_factor,
                      img_size=args.img_size,
                      batch_size=args.batch_size)

    elif (args.module == 'swarm'):
        run_swarm(nr_clients=args.nr_clients,
                  nr_local_epochs=args.nr_local_epochs,
                  nr_global_rounds=args.nr_global_rounds,
                  subset_factor=args.subset_factor,
                  img_size=args.img_size,
                  batch_size=args.batch_size)
    else:
        run_centralized(nr_clients=args.nr_clients,
                        nr_local_epochs=args.nr_local_epochs,
                        subset_factor=args.subset_factor,
                        img_size=args.img_size,
                        batch_size=args.batch_size,
                        tb_path=TB_PATH,
                        centralized_subpath=TB_CENTRALIZED_SUB_PATH)

        run_federated(nr_clients=args.nr_clients,
                      nr_local_epochs=args.nr_local_epochs,
                      nr_global_rounds=args.nr_global_rounds,
                      subset_factor=args.subset_factor,
                      img_size=args.img_size,
                      batch_size=args.batch_size,
                      tb_path=TB_PATH)


if __name__ == "__main__":
    main()
