import numpy as np
from edge_code.paramiko_ssh import SSHClient
import os
import sys
from edge_code.data_loader import load_datasets
from common.utilities import train, use_gpu, net_instance, get_parameters, set_parameters
from common.static_params import *
from flwr.common.logger import log
from logging import INFO

if __name__ == '__main__':
    # use gpu on edge devices
    use_gpu()
    # Get the command-line arguments
    cid = sys.argv[1]

    log(INFO,'download current model from server')
    
    with SSHClient(hostname=VM_IP, private_key_path=VM_KEY_PATH) as ssh:
        ssh.download_file("/root/Fleet/fleet-learning/tmp/agg.npz", "agg.npz")
    parameters = list(np.load("agg.npz",allow_pickle = True)['arr_0'])
    os.remove("agg.npz")
    model = net_instance(f"{cid}")
    set_parameters(model, parameters)

    log(INFO,'current model downloaded')

    log(INFO,f"load the data partition for client cid {cid}")
    
    with SSHClient(hostname=VM_IP, private_key_path=VM_KEY_PATH) as ssh:
        ssh.download_file("/root/Fleet/fleet-learning/tmp/partitions.npz", "partitions.npz")
    partition = np.load("partitions.npz")[cid]
    
    log(INFO,'data partition loaded')
    
    log(INFO,'load dataset to ram')
    
    trainloader, valloader, testloader = load_datasets(partition)
    
    log(INFO,'train the model')
    
    losses, accs, val_losses, val_accs = train(model, trainloader, valloader, NUM_LOCAL_EPOCHS, plot = False,client_cid = cid)
    
    log(INFO,f"loss: {losses}, accs: {accs}, val_losses: {val_losses}, val_accs: {val_accs}")
    
    # save the model
    params = get_parameters(model)
    params = np.array(params, dtype=object)
    np.savez("tmp/res"+cid+".npz", params)
    
    log(INFO,'upload model to server')
    with SSHClient(hostname=VM_IP, private_key_path=VM_KEY_PATH) as ssh:
        ssh.upload_file("tmp/res"+cid+".npz", "/root/Fleet/fleet-learning/tmp/res"+cid+".npz")

    #remove the model
    os.remove("tmp/res"+cid+".npz")
    log(INFO,'done')