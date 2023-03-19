import importlib
import importlib
import logging
from pathlib import Path
import numpy as np

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ..models.model_manager import load_model
from .utils import save_model, train, test

logger = logging.getLogger(__name__)


def run_centralised(config: dict, data: object, log_dir=str) -> None:
    logger.info("Initialising centralised learning runtime")
    model = load_model(config)
    if config["central"]["use_gpu"] == "true":
        model.to("cuda")
        logger.debug("Using GPU")
    else:
        model.to("cpu")
        logger.debug("Using CPU")

    # Load the loss function
    try:
        loss_method = config["model"]["loss"]
        module = importlib.import_module("torch.nn")
        loss_function = getattr(module, loss_method)
        logger.debug("Using loss function: {}".format(loss_method))
    except Exception as e:
        logger.error("Error loading loss function: {}".format(e))
        logger.exception(e)
        raise e

    # Start the timer for training duration
    start_time = datetime.now()

    # Create a tensorboard writer
    writer = SummaryWriter(log_dir)

    # Train the model
    mean_epoch_loss, batch_losses, mean_validation_loss, network = train(
        network=model,
        trainloader=data.train.dataloader,
        valloader=data.val.dataloader,
        epochs=config["central"]["epochs"],
        loss_function=loss_function,
        writer=writer,
        writer_path="centralised/loss/",
    )
    # save the modellog_dir
    model_name = Path(log_dir, f"centralised_{config['model']['name']}")
    save_model(model, model_name)
    
    
    # Calculate the training duration
    end_time = datetime.now()
    soft_duration = str(end_time - start_time).split(".")[0]

    # Log that training is finished
    logger.info("Finished centralised training after: {}".format(soft_duration))

    # Test the model
    logger.info("Evaluating the model on the test set")
    losses = test(network, data.test.dataloader, loss_function)
    test_loss = np.mean(losses)

    # Final logging
    logger.info("Final mean training loss: {}".format(mean_epoch_loss))
    logger.info("Final mean validation loss: {}".format(mean_validation_loss))
    logger.info("Final mean test loss: {}".format(test_loss))
    logger.info("END OF CENTRALISED LEARNING")
