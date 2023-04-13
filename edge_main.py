import numpy as np
from edge_code.paramiko_ssh import SSHClient
import os
import sys
from edge_code.data_loader import load_datasets
from common.utilities import train, use_gpu, net_instance, get_parameters, set_parameters
from common.static_params import *

if __name__ == '__main__':
    # use gpu on edge devices
    use_gpu()
    # Get the command-line arguments
    cid = sys.argv[1]
    strategy = sys.argv[2]

    print('download current model from server')
    
    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.download_file("/root/Fleet/fleet-learning/tmp/agg.npz", "agg.npz")
    parameters = list(np.load("agg.npz",allow_pickle = True)['arr_0'])
    os.remove("agg.npz")
    model = net_instance(f"{cid}")
    set_parameters(model, parameters)

    print('current model downloaded')

    print(f"load the data partition for client cid {cid}")
    
    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.download_file("/root/Fleet/fleet-learning/tmp/partitions.npz", "partitions.npz")
    partition = np.load("partitions.npz")[cid]
    
    print('data partition loaded')
    
    print('load dataset to ram')
    
    trainloader, valloader, testloader = load_datasets(partition)
    
    print('train the model')
    
    losses, accs, val_losses, val_accs = train(model, trainloader, valloader, NUM_LOCAL_EPOCHS, plot = False,client_cid = cid)
    
    print(f"loss: {losses}, accs: {accs}, val_losses: {val_losses}, val_accs: {val_accs}")
    
    # save the model
    params = get_parameters(model)
    params = np.array(params, dtype=object)
    np.savez("tmp/res"+cid+".npz", params)
    
    print('upload model to server')
    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.upload_file("tmp/res"+cid+".npz", "/root/Fleet/fleet-learning/tmp/res"+cid+".npz")

    #remove the model
    os.remove("tmp/res"+cid+".npz")
    print('done')