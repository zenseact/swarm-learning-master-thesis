import importlib
import importlib
import logging
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp

from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.transforms import *

from ..models.model_manager import load_model
from .utils import save_model, train, test, get_best_device, train_fixmatchseg

logger = logging.getLogger(__name__)


def run_centralised(config: dict, data: object, log_dir=str) -> None:
    """
    Trains a PyTorch model using centralised learning on a given data object.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model and training.
        data (DataObject): A fedswarm DataObject that has all the relevant data.
        log_dir (str): The directory path to save the training logs and model checkpoints.

    Returns:
        None
    """
    logger.info("Initialising centralised learning runtime")
    # Extract the model name from the config and create an instance of it
    model = load_model(config)

    # Move the model to the correct device
    if config["central"]["use_gpu"]:
        device = get_best_device()
        logger.debug(f"Moving model to device: {device}")
        model.to(device)
    else:
        model.to("cpu")
        logger.debug("Using CPU")

    # Load the loss function
    # Does not create an instance of the loss, just gets the class
    # flag for success
    success = False
    if not config["model"]["name"] == "fixmatchseg":
        try:
            loss_method = config["model"]["loss"]
            module = importlib.import_module("torch.nn")
            loss_class = getattr(module, loss_method)
            logger.debug("Using loss function: {}".format(loss_method))
            success = True
        except Exception as e:
            logger.error("Error loading loss function: {}".format(e))
            raise e
    else:
        loss_class = smp.losses.DiceLoss

    # Load the optimiser function
    # Does not create an instance of the optimiser, just gets the class
    try:
        opt_method = config["model"]["optimiser"]
        module = importlib.import_module("torch.optim")
        optimiser_class = getattr(module, opt_method)
        logger.debug("Using optimiser function: {}".format(opt_method))
    except KeyError:
        logger.debug("No optimiser specified, using Adam")
        optimiser_class = torch.optim.Adam
    except Exception as e:
        logger.error("Error loading loss function: {}".format(e))
        logger.exception(e)
        raise e

    # Start the timer for training duration
    start_time = datetime.now()

    # Create a tensorboard writer
    writer = SummaryWriter(log_dir)

    # Optimiser args
    # Fetch the optimiser args from the config
    try:
        optimiser_args = config["model"]["optimiser_args"]
        logger.debug("Using optimiser args: {}".format(optimiser_args))
    except KeyError:
        # If no args are specified, use an empty dictionary
        optimiser_args = {}
        
    # check if unlabelled is none
    if len(data.train.unlabelled) == 0 :
        unlabelled_loader = []
    else:
        unlabelled_loader = data.train.unlabelled.dataloader

    # Train the model
    if config["model"]["name"] == "fixmatchseg":
        weak_augmentation = Compose(eval(config["data"]["weak_augmentation"]))
        strong_augmentation = Compose(eval(config["data"]["strong_augmentation"]))
        normalisation = Compose(eval(config["data"]["normalisation"]))
        (
            mean_epoch_loss,
            batch_losses,
            mean_validation_loss,
            network,
        ) = train_fixmatchseg(
            network=model,
            labelled_loader=data.train.labelled.dataloader,
            unlabelled_loader=unlabelled_loader,
            valloader=data.val.dataloader,
            epochs=config["central"]["epochs"],
            loss_class=loss_class,
            optimiser_class=optimiser_class,
            weak_augmentation=weak_augmentation,
            strong_augmentation=strong_augmentation,
            normalisation=normalisation,
            optimiser_args=optimiser_args,
            writer=writer,
            writer_path="centralised/loss/",
            tau=config["model"]["tau"],
        )
    else:
        mean_epoch_loss, batch_losses, mean_validation_loss, network = train(
            network=model,
            trainloader=data.train.dataloader,
            valloader=data.val.dataloader,
            epochs=config["central"]["epochs"],
            loss_class=loss_class,
            optimiser_class=optimiser_class,
            optimiser_args=optimiser_args,
            writer=writer,
            writer_path="centralised/loss/",
        )
    # save the modellog_dir
    model_name = Path(log_dir, f"centralised_{config['model']['name']}")
    save_model(network, model_name)

    # Calculate the training duration
    end_time = datetime.now()
    soft_duration = str(end_time - start_time).split(".")[0]

    # Log that training is finished
    logger.info("Finished centralised training after: {}".format(soft_duration))

    # Test the model
    if data.test is not None:
        logger.info("Evaluating the model on the test set")
        losses = test(network, data.test.dataloader, loss_class)
        test_loss = np.mean(losses)
    else:
        logger.info("No test set provided, skipping test evaluation")
        test_loss = None

    # Final logging
    logger.info("Final mean training loss: {}".format(mean_epoch_loss))
    logger.info("Final mean validation loss: {}".format(mean_validation_loss))
    logger.info("Final mean test loss: {}".format(test_loss))
    logger.info("END OF CENTRALISED LEARNING")
