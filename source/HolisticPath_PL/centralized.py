from datasets import * 
from models import *
from train import *

def simulate_centralized():

    # get loaders
    trainloaders, valloaders, testloader, completeTrainloader, completeValloader = ZODImporter().load_datasets(num_clients=1)

    # create model
    model = PT_Model.load_from_checkpoint(c('checkpoint_path')) if(c('start_from_checkpoint')) else PT_Model()

    # train supervised
    trainer = train(model, completeTrainloader, completeValloader, nr_epochs=c('num_local_epochs'))

    # validate 
    validate(trainer, model, completeValloader)

    # test
    test(trainer, model, testloader)

    writer.close()


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

simulate_centralized()
#pred()

