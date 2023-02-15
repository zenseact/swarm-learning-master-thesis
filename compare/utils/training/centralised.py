import importlib
import importlib
import logging
import json

from datetime import datetime
from utils import ZODHandler as Data
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

from utils import load_model
from utils.training import train


def run_centralized(config: dict, data: Data, log_dir=str) -> None:
    model = load_model(config)
    if config["central"]["use_gpu"] == "true":
        model.to("cuda")
        logging.info("Using GPU")
    else:
        model.to("cpu")
        logging.info("Using CPU")

    # Load the loss function
    loss_method = config["central"]["loss"]
    module = importlib.import_module("torch.nn")
    loss_function = getattr(module, loss_method)

    start_time = datetime.now()

    writer = SummaryWriter(log_dir)

    # Train the model
    train(
        network=model,
        trainloader=data.train.dataloader,
        valloader=data.val.dataloader,
        epochs=config["central"]["epochs"],
        loss_function=loss_function,
        writer=writer,
        writer_path="centralised/loss/",
    )

    end_time = datetime.now()

    soft_duration = str(end_time - start_time).split(".")[0]

    # Log
    logging.info("Finished centralised training after: {}".format(soft_duration))
