import logging
import numpy as np

from torch.nn import Module
from torch import no_grad, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)


def train(
    network: Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    loss_class: object,
    optimiser_class: object,
    optimiser_args: dict = None,
    writer: SummaryWriter = None,
    writer_path: str = "loss/undefined/",
    server_round: int = None,
    cid: str = None,
) -> None:
    # Set up the loss function and optimizer
    criterion = loss_class()
    optimizer = optimiser_class(network.model_parameters(), **optimiser_args)

    # Get the device that the model is on
    device = next(network.parameters()).device
    logger.debug("Using device: {}".format(device))

    # Adjust and offset the epoch and batch counters if we are resuming training
    # This is used for correct epoch and batch logging. Primarily useful when doing decentralised training
    # Example: global round 2 with 15 epochs per round. Then on the second global round, we say that we are on epoch 16
    if server_round:
        running_batch_index = 1 + (server_round - 1) * len(trainloader) * epochs
        epoch_start = (server_round - 1) * epochs
        epoch_end = epoch_start + epochs
        logger.debug("RESUME: Starting at batch {}".format(running_batch_index))
        logger.debug("RESUME: Starting at epoch {}".format(epoch_start))
    else:
        running_batch_index = 1
        epoch_start = 0
        epoch_end = epochs

    # Start the training loop
    for epoch in range(epoch_start, epoch_end):
        logger.info("Now starting epoch {}".format(epoch + 1))
        network.train()
        batch_losses = []
        # Start the batch loop
        for batch_index, (images, labels) in enumerate(trainloader):
            logger.debug(
                "Starting: Epoch {}, Batch {}".format(epoch + 1, batch_index + 1)
            )
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            # Write the loss to tensorboard
            if writer is not None:
                try:
                    writer.add_scalars(
                        writer_path + "batch",
                        {"train": loss.item()},
                        running_batch_index,
                    )
                    logger.debug(
                        "[TENSORBOARD] Added batch loss: {:.2f}".format(loss.item())
                    )
                except Exception as e:
                    logger.error(
                        "[TENSORBOARD] Error writing to tensorboard: {}".format(e)
                    )
                    logger.exception(e)
            running_batch_index += 1

        # Test the model on the validation set
        validation_losses = test(
            network=network,
            dataloader=valloader,
            loss_class=loss_class,
        )
        # Calculate the mean losses
        mean_validation_loss = np.mean(validation_losses)
        mean_epoch_loss = np.mean(batch_losses)
        logger.info("Validation loss: {:.2f}".format(mean_validation_loss))
        logger.info("Training loss: {:.2f}".format(mean_epoch_loss))

        # Write the mean losses to tensorboard
        if writer is not None:
            try:
                writer.add_scalars(
                    writer_path + "epoch",
                    {"train": mean_epoch_loss, "validation": mean_validation_loss},
                    epoch + 1,
                )
                logger.debug(
                    "[TENSORBOARD] Added epoch loss: {:.2f}".format(mean_epoch_loss)
                )
            except Exception as e:
                logger.error("[TENSORBOARD] Error writing to tensorboard: {}".format(e))
                logger.exception(e)
    return mean_epoch_loss, batch_losses, mean_validation_loss, network


def test(network: Module, dataloader: DataLoader, loss_class: object):
    # Evaluate the network on the full content of the dataloader
    criterion = loss_class()
    device = next(network.parameters()).device
    network.eval()

    loss = []
    logger.debug("Starting evaluation on {} samples".format(len(dataloader.dataset)))
    start_time = datetime.now()
    with no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images).squeeze()
            loss.append(criterion(outputs, labels).item())
    end_time = datetime.now()
    soft_duration = str(end_time - start_time).split(".")[0]
    logger.debug("Finished evaluation after: {}".format(soft_duration))
    return loss


def save_model(model, name):
    try:
        save(model.state_dict(), f"{name}.pth")
        logger.info("Model saved as %s.pth", name)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        logger.exception(e)
