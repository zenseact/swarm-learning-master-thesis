from models import *
from train import *
from datasets import *

def run_supervised():

    zodImporter = ZODImporter()

    # get loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zodImporter.load_datasets(num_clients=1)

    # create model
    model = PTModel("FPN", "resnet34", in_channels=3, out_classes=1)

    # train supervised
    trainer = train(model, completeTrainloader, completeValloader, nr_epochs=NUM_LOCAL_EPOCHS)

    # validate 
    validate(trainer, model, completeValloader)

    # test
    test(trainer, model, testloader)


def pred():
    zodImporter = ZODImporter()

    # get loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zodImporter.load_datasets(num_clients=1)

    # load saved model
    checkpoint_path = "checkpoints/lightning_logs/version_0/checkpoints/epoch=2-step=21.ckpt"
    model = PTModel.load_from_checkpoint(checkpoint_path=checkpoint_path, arch="FPN", encoder_name="resnet34", in_channels=3, out_classes=1)

    # visualize samples from the dataset
    dataset_visualize(completeTrainloader.dataset, completeValloader.dataset, testloader.dataset)

    # predict and visualize predictions
    predict_visualize(model, testloader)

run_supervised()
#pred()