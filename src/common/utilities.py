from common.static_params import TASK, global_configs
from common.models import PTNet, Net

from collections import OrderedDict
import numpy as np
import torch
import gc
import time
from matplotlib import pyplot as plt
from typing import List
from common.logger import fleet_log
from logging import INFO

# from torch.utils.tensorboard import SummaryWriter


def train(
    net,
    trainloader,
    valloader,
    epochs: int = global_configs.NUM_LOCAL_EPOCHS,
):
    net.train()
    opt = torch.optim.Adam(net.model_parameters())
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(1, epochs + 1):
        tstart = time.time()
        batch_train_losses = []
        for data, target in trainloader:
            data, target = data.to(global_configs.DEVICE), target.to(
                global_configs.DEVICE
            )
            opt.zero_grad()
            output = net(data)
            loss = net.loss_fn(output, target)
            loss.backward()
            opt.step()

            batch_train_losses.append(loss.item())
        epoch_train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
        epoch_val_losses.append(test(net, valloader))
        print(
            f"[AGX] Epcoh completed in {time.time()-tstart:.2f} seconds with {len(trainloader)} batches of batch size {global_configs.BATCH_SIZE}"
        )
        print(f"[AGX] Train loss for epoch {epoch}: {epoch_train_losses[-1]:.2f}")
        print(f"[AGX] Validation loss for epoch {epoch}: {epoch_val_losses[-1][0]:.2f}")
    return epoch_train_losses


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = (
        torch.nn.CrossEntropyLoss()
        if global_configs.ML_TASK == TASK.CLASSIFICATION
        else torch.nn.MSELoss()
    )
    correct, total = 0, 0
    net.eval()
    loss = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(global_configs.DEVICE), labels.to(
                global_configs.DEVICE
            )
            outputs = net(images)
            outputs = outputs.unsqueeze(0) if outputs.shape[0] != 1 else outputs
            labels = labels.unsqueeze(0) if labels.shape[0] != 1 else labels
            loss.append(criterion(outputs, labels).item())

            if global_configs.ML_TASK == TASK.CLASSIFICATION:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    loss = np.mean(loss)
    accuracy = (
        correct / total if global_configs.ML_TASK == TASK.CLASSIFICATION else None
    )
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    if global_configs.PRINT_DEBUG_DATA:
        fleet_log(INFO, "⤺ Get model parameters")
    net = net.to("cpu")
    params = [val.numpy() for _, val in net.state_dict().items()]
    net = net.to(global_configs.DEVICE)
    return params


def set_parameters(net, parameters: List[np.ndarray]):
    if global_configs.PRINT_DEBUG_DATA:
        fleet_log(INFO, "⤻ Set model parameters")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)


def print_gpu_processes(extra_info=None):
    try:
        if extra_info:
            fleet_log(INFO, extra_info, torch.cuda.list_gpu_processes())
        else:
            fleet_log(INFO, f"{torch.cuda.list_gpu_processes()}")
    except:
        pass


def clear_gpu():
    print_gpu_processes()
    fleet_log(INFO, "started clearing the GPU RAM.")
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        fleet_log(INFO, "Could not clear the GPU RAM.")
    print_gpu_processes()
    fleet_log(INFO, "Done clearing the GPU RAM.")


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
    net = (
        PTNet().to(global_configs.DEVICE)
        if global_configs.RUN_PRETRAINED
        else Net().to(global_configs.DEVICE)
    )
    fleet_log(INFO, f"🌻 Created new model - {name} 🌻")
    return net


def destroy_model(model, name):
    print_gpu_processes(f"⍤ [show gpu process usage before destroying model {name}] ⍤")
    model = model.cpu()
    # odel.destroy()
    del model
    fleet_log(INFO, f"꧁ destroyed model - {name} ꧂")
    print_gpu_processes(f"⍤ [show gpu process usage after destroying model {name}] ⍤")


def use_cpu(model=None):
    fleet_log(INFO, "swiched to cpu")
    DEVICE = torch.device("cpu")
    if model:
        model = model.cpu()


def use_gpu():
    fleet_log(INFO, "swiched to gpu")
    DEVICE = torch.device("cuda")


def plot_client_losses(clients_data):
    for client in clients_data:
        fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
        plot_metrics(
            [ax],
            [[client.losses, client.val_losses]],
            ["RMSE Loss"],
            ["Number of epochs"],
            [["Train", "Val"]],
        )


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
    return f"exec time: {round((end_time - start_time),3)} seconds"
