from fleetlearning.common.utilities import train, set_parameters
from fleetlearning.scheduler.data_loader import load_datasets


def train_model(message_from_virtual_vehicle, model, zod_frames):
    trainloader, valloader = load_datasets(
        message_from_virtual_vehicle["data"]["partitions"], zod_frames
    )
    set_parameters(model, message_from_virtual_vehicle["data"]["model"])
    _ = train(
        net=model,
        trainloader=trainloader,
        valloader=valloader,
    )
