from common.static_params import TASK, global_configs
from common.models import PTNet, Net

from collections import OrderedDict
import numpy as np
import torch
import gc
import time
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Tuple
from common.logger import fleet_log
from logging import INFO
# from torch.utils.tensorboard import SummaryWriter

def train(net, trainloader, valloader,
          epochs: int, contin_val=True, plot=True,
          client_cid=None, verbose=1, model_name="",
          tb_subpath=None, tb_writer=None, server_round=1):
    """Train the network on the training set."""
    fleet_log(INFO,
        f'⇉ Started transfer learning of model {model_name}' if net.is_pretrained else f'Started normal learning of model {model_name}')
    num_batches = len(trainloader)
    print_every = (num_batches // 3) if (num_batches // 3) != 0 else 1

    """tensor board publish"""
    running_batch_index = 1 + (server_round - 1) * len(trainloader)

    criterion = torch.nn.CrossEntropyLoss() if global_configs.ML_TASK == TASK.CLASSIFICATION else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.model_parameters())
    net.train()

    losses = []
    accs = []
    val_losses = []
    val_accs = []
    training_start_time = timer_start()
    for epoch in range(epochs):
        correct, total, epoch_acc, epoch_val_accuracy = 0, 0, 0.0, 0.0
        epoch_loss = []
        epoch_start_time = timer_start()
        batch_start_time = timer_start()
        for batch_index, (images, labels) in enumerate(trainloader):
            images, labels = images.to(global_configs.DEVICE), labels.to(global_configs.DEVICE)
            optimizer.zero_grad()
            outputs = net(images)

            outputs = outputs.unsqueeze(0) if outputs.shape[0] != 1 else outputs
            labels = labels.unsqueeze(0) if labels.shape[0] != 1 else labels

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss.append(loss.item())
            total += labels.size(0)

            if tb_writer is not None:
                tb_writer.add_scalars(
                    tb_subpath + "batch",
                    {"train": loss.item()},
                    running_batch_index,
                )
            running_batch_index += 1

            if (global_configs.ML_TASK == TASK.CLASSIFICATION):
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
            if batch_index % print_every == 0 and verbose > 0:
                fleet_log(INFO,f"\tBatch {batch_index}/{num_batches}: Train loss: {sum(epoch_loss[-print_every:]) / print_every:.3f}, {timer_end(batch_start_time)}")
                batch_start_time = timer_start()

        epoch_loss = np.mean(epoch_loss)
        epoch_val_loss, epoch_val_accuracy = test(net, valloader)
        losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)

        if tb_writer is not None:
            tb_writer.add_scalars(
                tb_subpath + "epoch",
                {"train": np.mean(epoch_loss), "validation": np.mean(epoch_val_loss)},
                (server_round - 1) * epochs + epoch + 1,
            )

        if (global_configs.ML_TASK == TASK.CLASSIFICATION):
            epoch_acc = correct / total
            accs.append(epoch_acc)
            val_accs.append(epoch_val_accuracy)

            fleet_log(INFO,
                f" ↪ Client{client_cid} Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}, val loss {epoch_val_loss}, accuracy {epoch_val_accuracy}")
        else:
            fleet_log(INFO,f" ↪ Client{client_cid} Epoch {epoch + 1}: train loss {epoch_loss},\t val loss {epoch_val_loss},\t {timer_end(epoch_start_time)}")

    fleet_log(INFO,f'Complete Training {timer_end(training_start_time)}')

    if (plot):
        if (global_configs.ML_TASK == TASK.CLASSIFICATION):
            fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
            plot_metrics(ax, [[losses, val_losses], [accs, val_accs]],
                         [f'Corss Entropy Loss - model {model_name}', f'Accuracy - model {model_name}'],
                         ['Number of epochs', 'Number of epochs'], [['Train', 'Val'], ['Train', 'Val']])
        else:
            fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
            plot_metrics([ax], [[losses, val_losses]], [f'RMSE Loss - model {model_name}'],
                         [f'Number of epochs - model {model_name}'], [['Train', 'Val']])

    fleet_log(INFO,"For manual plotting:")
    fleet_log(INFO,f"Client{client_cid} Train losses = {losses}")
    fleet_log(INFO,f"Client{client_cid} Val_losses = {val_losses}")
    return losses, accs, val_losses, val_accs


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss() if global_configs.ML_TASK == TASK.CLASSIFICATION else torch.nn.MSELoss()
    correct, total = 0, 0
    net.eval()
    loss = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(global_configs.DEVICE), labels.to(global_configs.DEVICE)
            outputs = net(images)
            outputs = outputs.unsqueeze(0) if outputs.shape[0] != 1 else outputs
            labels = labels.unsqueeze(0) if labels.shape[0] != 1 else labels
            loss.append(criterion(outputs, labels).item())

            if (global_configs.ML_TASK == TASK.CLASSIFICATION):
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    loss = np.mean(loss)
    accuracy = correct / total if global_configs.ML_TASK == TASK.CLASSIFICATION else None
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    if (global_configs.PRINT_DEBUG_DATA): fleet_log(INFO,"⤺ Get model parameters")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    if (global_configs.PRINT_DEBUG_DATA): fleet_log(INFO,"⤻ Set model parameters")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def print_gpu_processes(extra_info=None):
    try:
        if (extra_info):
            fleet_log(INFO,extra_info, torch.cuda.list_gpu_processes())
        else:
            fleet_log(INFO,f"{torch.cuda.list_gpu_processes()}")
    except:
        pass


def clear_gpu():
    print_gpu_processes()
    fleet_log(INFO,"started clearing the GPU RAM.")
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        fleet_log(INFO,"Could not clear the GPU RAM.")
    print_gpu_processes()
    fleet_log(INFO,"Done clearing the GPU RAM.")


def plot_metrics(axs, metrics, titles, xlabels, legends, ylim=None):
    for i, ax in enumerate(axs):
        for m in metrics[i]:
            ax.plot(m)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabels[i])
        ax.legend(legends[i])
        if ylim:
            ax.set_ylim(ylim)


def net_instance(name):
    net = PTNet().to(global_configs.DEVICE) if global_configs.RUN_PRETRAINED else Net().to(global_configs.DEVICE)
    fleet_log(INFO,f"🌻 Created new model - {name} 🌻")
    return net


def destroy_model(model, name):
    print_gpu_processes(f"⍤ [show gpu process usage before destroying model {name}] ⍤")
    model = model.cpu()
    # odel.destroy()
    del model
    fleet_log(INFO,f"꧁ destroyed model - {name} ꧂")
    print_gpu_processes(f"⍤ [show gpu process usage after destroying model {name}] ⍤")


def use_cpu(model=None):
    fleet_log(INFO,"swiched to cpu")
    DEVICE = torch.device('cpu')
    if (model):
        model = model.cpu()


def use_gpu():
    fleet_log(INFO,"swiched to gpu")
    DEVICE = torch.device('cuda')


def plot_client_losses(clients_data):
    for client in clients_data:
        fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
        plot_metrics([ax], [[client.losses, client.val_losses]], ['RMSE Loss'], ['Number of epochs'],
                     [['Train', 'Val']])


def save_dataset_tb_plot(tb_path, sample_distribution, subtitle, seed):
    pass
    # plt.bar(list(range(1, len(sample_distribution) + 1)), sample_distribution)
    # plt.xlabel("Partitions")
    # plt.ylabel("Samples")
    # plt.suptitle("Distribution of samples")
    # plt.title("%s, seed: %s" % (subtitle, seed)),

    # """report to tensor board"""
    # writer = SummaryWriter(tb_path)
    # writer.add_figure("sample_distribution/%s" % (subtitle), plt.gcf(), global_step=0)
    # writer.close()

def timer_start():
    start_time = time.time()
    return start_time

def timer_end(start_time):
    end_time = time.time()
    return f'exec time: {round((end_time - start_time),3)} seconds'