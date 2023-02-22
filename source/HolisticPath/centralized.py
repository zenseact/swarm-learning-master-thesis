from utilities import *
from datasets import *


class CentralizedSimulator:

    def __init__(self, trainloaders, valloaders, testloader, device=DEVICE, tb_path=None, centralized_subpath=None):
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloader = testloader
        self.device = device
        self.tb_path = tb_path
        self.centralized_subpath=centralized_subpath

    def sim_cen(self, print_summery=False, nr_local_epochs=NUM_LOCAL_EPOCHS):
        # create the net
        net = net_instance("Centralized")

        # data
        trainloader = self.trainloaders[0]
        valloader = self.valloaders[0]

        # summery
        print('nr of training imgs:', len(trainloader.dataset))
        print('nr of validation imgs:', len(valloader.dataset))
        print('nr of test imgs:', len(self.testloader.dataset))
        print('input shape:', trainloader.dataset[0][0].shape)
        print('output shape:', trainloader.dataset[0][1].shape)
        print(f'training on {self.device}')
        if (print_summery):
            print(summary(net, trainloader.dataset[0][0].shape))

        writer = SummaryWriter(self.tb_path)

        # train & val
        train(
            net,
            trainloader,
            valloader,
            epochs=nr_local_epochs,
            contin_val=True,
            plot=True,
            verbose=0,
            model_name=f"Centralized",
            tb_subpath=self.centralized_subpath,
            tb_writer=writer,
            server_round=1
        )
        loss, accuracy = test(net, self.testloader)
        if (ML_TASK == TASK.CLASSIFICATION):
            print(f"►►► test loss {loss}, accuracy {accuracy}")
        else:
            print(f"►►► test RMSE {loss}")

        writer.close()
        return float(loss), len(valloader), {"accuracy": float(accuracy) if accuracy else None}


def main(
        nr_clients=2,
        nr_local_epochs=2,
        subset_factor=SUBSET_FACTOR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        tb_path=TB_PATH,
        centralized_subpath=TB_CENTRALIZED_SUB_PATH):

    # import Zod data into memory
    zod = ZODImporter(subset_factor=subset_factor, img_size=img_size, batch_size=batch_size, tb_path=tb_path)

    # create pytorch loaders
    trainloaders, valloaders, testloader = zod.load_datasets(nr_clients)

    # create federated simulator
    cen_sim = CentralizedSimulator(trainloaders, valloaders, testloader, device, tb_path, centralized_subpath)

    # simulate federated learning
    cen_sim.sim_cen(print_summery=False, nr_local_epochs=nr_local_epochs)


if __name__ == "__main__":
    main()
