import logging
from pathlib import Path
import numpy as np
import torch
import GPUtil
import itertools

from torch.nn import Module
from torch import no_grad, save
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import segmentation_models_pytorch as smp
import torch.nn.functional as F

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
    best_validation_loss = 999999
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
            # reshape the target tensor to match the input tensor
            labels = labels.view_as(outputs)
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
        if "clients" in writer_path:
            name_addition = "client_" + str(
                writer_path.split("clients")[-1].replace("/", "")
            )
            name = f"{writer.get_logdir()}/best_model_{name_addition}"
        else:
            name = f"{writer.get_logdir()}/best_model"
        validation_losses = test(
            network=network,
            dataloader=valloader,
            loss_class=loss_class,
            best_loss=best_validation_loss,
            epoch=epoch,
            name=name,
        )
        # Calculate the mean losses
        mean_validation_loss = np.mean(validation_losses)
        best_validation_loss = min(best_validation_loss, mean_validation_loss)
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


def train_fixmatchseg(
    network: Module,
    labelled_loader: DataLoader,
    unlabelled_loader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    loss_class: object,
    tau: float,
    optimiser_class: object,
    weak_augmentation: callable,
    strong_augmentation: callable,
    normalisation: callable,
    optimiser_args: dict = None,
    writer: SummaryWriter = None,
    writer_path: str = "loss/undefined/",
    server_round: int = None,
    cid: str = None,
) -> None:
    # Display tau
    logger.info("Using tau: {}".format(tau))
    
    # Check if unlabelled loader is None
    if len(unlabelled_loader) == 0:
        logger.info("No unlabelled data provided. Using supervised only training")
        supervised_only = True
    else:
        supervised_only = False
    
    # Set up the loss function and optimizer
    optimizer = optimiser_class(network.model_parameters(), **optimiser_args)

    # Get the device that the model is on
    device = next(network.parameters()).device
    logger.debug("Using device: {}".format(device))

    # Adjust and offset the epoch and batch counters if we are resuming training
    # This is used for correct epoch and batch logging. Primarily useful when doing decentralised training
    # Example: global round 2 with 15 epochs per round. Then on the second global round, we say that we are on epoch 16
    
    if supervised_only:
        largest_loader = len(labelled_loader)
    else:
        largest_loader = max(len(labelled_loader), len(unlabelled_loader))
        labelled_size = len(labelled_loader)
    if server_round:
        running_batch_index = 1 + (server_round - 1) * largest_loader * epochs
        epoch_start = (server_round - 1) * epochs
        epoch_end = epoch_start + epochs
        logger.debug("RESUME: Starting at batch {}".format(running_batch_index))
        logger.debug("RESUME: Starting at epoch {}".format(epoch_start))
    else:
        running_batch_index = 1
        epoch_start = 0
        epoch_end = epochs

    # cycle the smallest loader
    if not supervised_only:
        if len(labelled_loader) > len(unlabelled_loader):
            unlabelled_loader = itertools.cycle(unlabelled_loader)
        else:
            labelled_loader = itertools.cycle(labelled_loader)

    # Start the training loop
    best_validation_loss = 999999
    for epoch in range(epoch_start, epoch_end):
        logger.info("Now starting epoch {}".format(epoch + 1))
        network.train()
        batch_losses = []
        supervised_losses = []
        unsupervised_losses = []

        # Start the batch loop for unsupervised and supervised
        if not supervised_only:
            for (
                batch_index,
                ((labelled_images, image_labels), (unlabelled_images, _)),
            ) in enumerate(zip(labelled_loader, unlabelled_loader)):
                logger.debug(
                    "Starting: Epoch {}, Batch {}".format(epoch + 1, batch_index + 1)
                )
                # reset gradients
                optimizer.zero_grad()
                # perform augmentation on cpu
                unlabelled_weakly_augmented_images = weak_augmentation(unlabelled_images)
                unlabelled_strongly_augmented_images = strong_augmentation(
                    unlabelled_weakly_augmented_images
                )
                # normalise the images on cpu
                normalised_labelled_images = normalisation(labelled_images)
                normalised_unlabelled_weakly_augmented_images = normalisation(
                    unlabelled_weakly_augmented_images
                )
                normalised_unlabelled_strongly_augmented_images = normalisation(
                    unlabelled_strongly_augmented_images
                )
                # move images to gpu
                input = torch.concat(
                    [
                        normalised_labelled_images,
                        normalised_unlabelled_weakly_augmented_images,
                        normalised_unlabelled_strongly_augmented_images,
                    ],
                    dim=0,
                )
                input = input.to(device)
                # move labels to gpu
                image_labels = image_labels.to(device)
                output = network(input)
                # split the output into the three parts
                shapes = [
                    labelled_images.shape[0],
                    unlabelled_weakly_augmented_images.shape[0],
                    unlabelled_strongly_augmented_images.shape[0],
                ]
                (
                    labelled_predictions,
                    unlabelled_weakly_augmented_predictions,
                    unlabelled_strongly_augmented_predictions,
                ) = torch.split(output, shapes, dim=0)
                # create the max value of the predictions and the psuedo labels
                # We do this because the predictions are sigmoided
                qb = torch.abs(unlabelled_weakly_augmented_predictions - 0.5) * 2
                max_labelled_confidences = torch.abs(labelled_predictions -0.5) * 2
                mean_labelled_confidence = torch.mean(max_labelled_confidences)
                # get binary labels
                qhatb = (unlabelled_weakly_augmented_predictions > 0.5).float()
                # calculate the mean of the max values
                qb_bar = torch.mean(qb, dim=[1, 2, 3])
                # get indicies of psuedo labels
                psuedo_indicies = torch.nonzero(qb_bar > tau)
                psuedo_labels = qhatb[psuedo_indicies]
                selected_unlabelled_strongly_augmented_predictions = (
                    unlabelled_strongly_augmented_predictions[psuedo_indicies]
                )
                # sanity checks
                logger.debug(
                    f"Weak/Strong equal: {(unlabelled_strongly_augmented_predictions == unlabelled_weakly_augmented_predictions).all()}"
                )
                logger.debug(
                    f"Psuedo/Strong equal (selected): {(selected_unlabelled_strongly_augmented_predictions == psuedo_labels).all()}"
                )
                # calculate the loss
                supervised_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)(
                    labelled_predictions, image_labels
                )
                if len(psuedo_indicies) > 0:
                    unsupervised_loss = smp.losses.DiceLoss(
                        mode="binary", from_logits=False
                    )(selected_unlabelled_strongly_augmented_predictions, psuedo_labels)
                else:
                    unsupervised_loss = torch.tensor(0)
                # combine the losses
                loss = supervised_loss + unsupervised_loss
                loss.backward()
                # update the weights
                optimizer.step()

                mean_qb_bar = torch.mean(qb_bar)
                max_qb_bar = torch.max(qb_bar)
                min_qb_bar = torch.min(qb_bar)
                batch_losses.append(loss.item())
                supervised_losses.append(supervised_loss.item())
                unsupervised_losses.append(unsupervised_loss.item())
                passed_unlabelled = len(psuedo_indicies)
                # Write the loss to tensorboard
                if writer is not None:
                    try:
                        writer.add_scalars(
                            writer_path + "batch",
                            {
                                "train": loss.item(),
                                "supervised": supervised_loss.item(),
                                "unsupervised": unsupervised_loss.item(),
                            },
                            running_batch_index,
                        )
                        writer.add_scalars(
                            writer_path + "internal_counters",
                            {
                                "utilised labelled": len(labelled_images),
                                "utilised unlabelled": passed_unlabelled,
                            },
                            running_batch_index,
                        )
                        writer.add_scalars(
                            writer_path + "dataloader_counters",
                            {
                                "labelled batch index": batch_index % labelled_size,
                                "unlabelled batch index": batch_index,
                            },
                            running_batch_index,
                        )
                        writer.add_scalars(
                            writer_path + "internal_variables",
                            {
                                "qb_bar_mean": mean_qb_bar,
                                "qb_bar_max": max_qb_bar,
                                "qb_bar_min": min_qb_bar,
                                "tau": tau,
                                "mean_labelled_confidence": mean_labelled_confidence,
                            },
                            running_batch_index,
                        )
                        logger.debug(
                            "[TENSORBOARD] Added batch loss: {:.2f}".format(loss.item())
                        )
                        logger.debug(
                            "[TENSORBOARD] Added supervised loss: {:.2f}".format(
                                supervised_loss.item()
                            )
                        )
                        logger.debug(
                            "[TENSORBOARD] Added unsupervised loss: {:.2f}".format(
                                unsupervised_loss.item()
                            )
                        )
                        logger.debug(
                            "[TENSORBOARD] Using: {:.2f} psuedo labels".format(
                                passed_unlabelled
                            )
                        )
                        logger.debug(
                            "[TENSORBOARD] Using: {:.2f} labelled frames".format(
                                len(labelled_images)
                            )
                        )
                        logger.debug(
                            "[TENSORBOARD] qb_bar_mean: {:.2f}, qb_bar_max: {:.2f}, qb_bar_min: {:.2f}, mean_labelled_confidence {:.2f}".format(
                                mean_qb_bar, max_qb_bar, min_qb_bar, mean_labelled_confidence
                            )
                        )
                    except Exception as e:
                        logger.error(
                            "[TENSORBOARD] Error writing to tensorboard: {}".format(e)
                        )
                        logger.exception(e)
                running_batch_index += 1
        else:
            for (
                batch_index,
                (labelled_images, image_labels),
            ) in enumerate(labelled_loader):
                logger.debug(
                    "Starting: Epoch {}, Batch {}".format(epoch + 1, batch_index + 1)
                )
                # reset gradients
                optimizer.zero_grad()
                # normalise the images on cpu
                normalised_labelled_images = normalisation(labelled_images)
                input = normalised_labelled_images.to(device)
                # move labels to gpu
                image_labels = image_labels.to(device)
                labelled_predictions = network(input)
                max_labelled_confidences = torch.abs(labelled_predictions -0.5) * 2
                mean_labelled_confidence = torch.mean(max_labelled_confidences)
                # calculate the loss
                supervised_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)(
                    labelled_predictions, image_labels
                )
                loss = supervised_loss
                loss.backward()
                # update the weights
                optimizer.step()
                batch_losses.append(loss.item())
                # Write the loss to tensorboard
                if writer is not None:
                    try:
                        writer.add_scalars(
                            writer_path + "batch",
                            {
                                "train": loss.item(),
                            },
                            running_batch_index,
                        )
                        writer.add_scalars(
                            writer_path + "internal_counters",
                            {
                                "utilised labelled": len(labelled_images)
                            },
                            running_batch_index,
                        )
                        writer.add_scalars(
                            writer_path + "internal_variables",
                            {
                                "mean_labelled_confidence": mean_labelled_confidence,
                            },
                            running_batch_index,
                        )
                        logger.debug(
                            "[TENSORBOARD] Added batch loss: {:.2f}".format(loss.item())
                        )
                        logger.debug(
                            "[TENSORBOARD] Using: {:.2f} labelled frames".format(
                                len(labelled_images)
                            )
                        )
                        logger.debug(
                            "[TENSORBOARD] mean_labelled_confidence {:.2f}".format(mean_labelled_confidence)
                        )
                    except Exception as e:
                        logger.error(
                            "[TENSORBOARD] Error writing to tensorboard: {}".format(e)
                        )
                        logger.exception(e)
                running_batch_index += 1
        # Test the model on the validation set
        if "clients" in writer_path:
            name_addition = "client_" + str(
                writer_path.split("clients")[-1].replace("/", "")
            )
            name = f"{writer.get_logdir()}/best_model_{name_addition}"
        else:
            name = f"{writer.get_logdir()}/best_model"
        validation_losses = test(
            network=network,
            dataloader=valloader,
            loss_class=smp.losses.DiceLoss,
            best_loss=best_validation_loss,
            epoch=epoch,
            name=name,
        )
        # Calculate the mean losses
        mean_validation_loss = np.mean(validation_losses)
        best_validation_loss = min(best_validation_loss, mean_validation_loss)
        mean_epoch_loss = np.mean(batch_losses)
        logger.info("Validation loss: {:.2f}".format(mean_validation_loss))
        logger.info("Training loss: {:.2f}".format(mean_epoch_loss))

        # Write the mean losses to tensorboard
        if writer is not None:
            try:
                writer.add_scalars(
                    writer_path + "epoch",
                    {
                        "train": mean_epoch_loss,
                        "validation": mean_validation_loss,
                        "supervised": np.mean(supervised_losses),
                        "unsupervised": np.mean(unsupervised_losses),
                    },
                    epoch + 1,
                )
                logger.debug(
                    "[TENSORBOARD] Added epoch loss: {:.2f}, supervised: {:.2f}, unsupervised: {:.2f}".format(
                        mean_epoch_loss,
                        np.mean(supervised_losses),
                        np.mean(unsupervised_losses),
                    )
                )
            except Exception as e:
                logger.error("[TENSORBOARD] Error writing to tensorboard: {}".format(e))
                logger.exception(e)
    return mean_epoch_loss, batch_losses, mean_validation_loss, network


def test(
    network: Module,
    dataloader: DataLoader,
    loss_class: object,
    best_loss: float = None,
    epoch: int = None,
    name=None,
):
    # Evaluate the network on the full content of the dataloader
    if loss_class == smp.losses.DiceLoss:
        criterion = loss_class(mode="binary", from_logits=False)
    else:
        criterion = loss_class()
    device = next(network.parameters()).device
    network.eval()

    loss = []
    logger.debug("Starting evaluation on {} samples".format(len(dataloader.dataset)))
    start_time = datetime.now()
    with no_grad():
        for images, labels in dataloader:
            logger.debug(
                "Evaluating batch {} of {}".format(len(loss) + 1, len(dataloader))
            )
            images, labels = images.to(device), labels.to(device)
            outputs = network(images).squeeze()
            labels = labels.view_as(outputs)
            loss.append(criterion(outputs, labels).item())
    end_time = datetime.now()
    soft_duration = str(end_time - start_time).split(".")[0]
    logger.debug("Finished evaluation after: {}".format(soft_duration))

    if best_loss is not None and name:
        if np.mean(loss) <= best_loss:
            logger.info("New best model found. Saving...")
            save_model(network, name)
            with open(f"{name}.txt", "w") as f:
                f.write("Epoch: {}\n".format(epoch))
                f.write("Loss: {:.2f}\n".format(np.mean(loss)))

    return loss


def save_model(model, name):
    try:
        save(model.state_dict(), f"{name}.pth")
        logger.info("Model saved as %s.pth", name)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        logger.exception(e)


def get_best_device():
    if torch.cuda.is_available():
        GPUs = GPUtil.getGPUs()
        gpu_availability = [gpu.memoryFree for gpu in GPUs]
        best_gpu_index = gpu_availability.index(max(gpu_availability))
        return torch.device(f"cuda:{best_gpu_index}")
    else:
        return torch.device("cpu")
