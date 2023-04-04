from utilities import *
from datasets import *


class CentralizedSimulator:
    def __init__(
        self,
        trainloader,
        valloader,
        testloader,
        device=DEVICE,
        tb_path=None,
        centralized_subpath=None,
    ):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.tb_path = tb_path
        self.centralized_subpath = centralized_subpath

    def sim_cen(self, print_summery=False, nr_local_epochs=NUM_LOCAL_EPOCHS):
        # create the net
        net = net_instance("Centralized")

        # summery
        print("nr of training imgs:", len(self.trainloader.dataset))
        print("nr of validation imgs:", len(self.valloader.dataset))
        print("nr of test imgs:", len(self.testloader.dataset))
        print("input shape:", self.trainloader.dataset[0][0].shape)
        print("output shape:", self.trainloader.dataset[0][1].shape)
        print(f"training on {self.device}")
        if print_summery:
            print(summary(net, self.trainloader.dataset[0][0].shape))

        writer = SummaryWriter(self.tb_path)

        # train & val
        train(
            net,
            self.trainloader,
            self.valloader,
            epochs=nr_local_epochs,
            contin_val=True,
            plot=True,
            verbose=1,
            model_name=f"Centralized",
            tb_subpath=self.centralized_subpath,
            tb_writer=writer,
            server_round=1,
            client_cid="Centralized",
        )
        loss, accuracy = test(net, self.testloader)
        if ML_TASK == TASK.CLASSIFICATION:
            print(f"►►► test loss {loss}, accuracy {accuracy}")
        else:
            print(f"►►► test RMSE {loss}")

        writer.close()
        return (
            float(loss),
            len(self.valloader),
            {"accuracy": float(accuracy) if accuracy else None},
        )


def main(
    nr_clients=2,
    nr_local_epochs=4,
    subset_factor=0.05,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    tb_path=TB_PATH,
    centralized_subpath=TB_CENTRALIZED_SUB_PATH,
):
    # import Zod data into memory
    zod = ZODImporter(
        subset_factor=subset_factor,
        img_size=img_size,
        batch_size=batch_size,
        tb_path=tb_path,
        stored_gt_path=STORED_GROUND_TRUTH_PATH,
        stored_balanced_ds_path=None,
    )  # STORED_GROUND_TRUTH_PATH, STORED_BALANCED_DS_PATH

    # create pytorch loaders
    (
        trainloaders,
        valloaders,
        testloader,
        completeTrainloader,
        completeValloader,
    ) = zod.load_datasets(nr_clients)

    # create federated simulator
    cen_sim = CentralizedSimulator(
        completeTrainloader,
        completeValloader,
        testloader,
        device,
        tb_path,
        centralized_subpath,
    )

    # simulate federated learning
    cen_sim.sim_cen(print_summery=False, nr_local_epochs=nr_local_epochs)


if __name__ == "__main__":
    main()
