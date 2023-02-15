import logging
import numpy as np

from torch.nn import Module
from torch import no_grad, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def train(
    network: Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    loss_function: type,
    writer: SummaryWriter = None,
    writer_path: str = "loss/undefined/",
    server_round: int = None,
) -> None:
    criterion = loss_function()
    optimizer = Adam(network.model_parameters())
    device = next(network.parameters()).device

    if server_round:
        running_batch_index = 1 + (server_round - 1) * len(trainloader)
        epoch_start = (server_round - 1) * epochs
        epoch_end = epoch_start + epochs
    else:
        running_batch_index = 1
        epoch_start = 0
        epoch_end = epochs

    for epoch in range(epoch_start, epoch_end):
        logging.info("Epoch %d", epoch + 1)
        network.train()
        epoch_loss = []
        for batch_index, (images, labels) in enumerate(trainloader):
            logging.debug("Batch %d", batch_index + 1)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalars(
                    writer_path + "batch",
                    {"train": loss.item()},
                    running_batch_index,
                )
            running_batch_index += 1
        test_loss = test(
            network=network,
            dataloader=valloader,
            loss_function=loss_function,
        )
        logging.debug("Validation loss: %d", test_loss)
        logging.debug("Training loss: %d", np.mean(epoch_loss))
        if writer is not None:
            writer.add_scalars(
                writer_path + "epoch",
                {"train": np.mean(epoch_loss), "validation": np.mean(test_loss)},
                epoch + 1,
            )
    return np.mean(epoch_loss), epoch_loss


def test(network: Module, dataloader: DataLoader, loss_function: type):
    """Evaluate the network on the entire test set."""
    criterion = loss_function()
    device = next(network.parameters()).device
    network.eval()

    loss = []
    with no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            loss.append(criterion(outputs, labels).item())
    return loss


# TODO: Re-implement this function so that we can use it again (cl,fl,sl)
def save_model(model, name):
    logging.debug("Save model %s", name)
    save(model.state_dict(), f"{name}.pth")
