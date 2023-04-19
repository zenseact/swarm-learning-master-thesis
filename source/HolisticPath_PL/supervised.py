from datasets import * 
from models import *

from models import *
from train import *
from datasets import *

def run_supervised():

    zodImporter = ZODImporter()

    # get loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zodImporter.load_datasets(num_clients=1)

    # create model
    model = PT_Model()

    # train supervised
    trainer = train(model, completeTrainloader, completeValloader, nr_epochs=NUM_LOCAL_EPOCHS)

    # validate 
    validate(trainer, model, completeValloader)

    # test
    #test(trainer, model, testloader)


def pred():
    zodImporter = ZODImporter()

    # get loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = zodImporter.load_datasets(num_clients=1)

    # load saved model
    checkpoint_path = "checkpoints/lightning_logs/version_0/checkpoints/epoch=2-step=21.ckpt"
    model = PT_Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # pred
    frame_id  =list(completeTrainloader)[0]
    pred = predict(model.to(DEVICE), zodImporter.zod_frames, frame_id)

    # visualize
    visualize_HP_on_image(zodImporter.zod_frames, frame_id, pred)

run_supervised()
#pred()

