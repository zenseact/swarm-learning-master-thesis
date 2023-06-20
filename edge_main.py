import numpy as np
from edge_code.paramiko_ssh import SSHClient
import os
import sys
from edge_code.data_loader import load_datasets
from common.utilities import train, use_gpu, net_instance, get_parameters, set_parameters
from common.static_params import global_configs
from common.logger import fleet_log
from logging import INFO

def main():
    # use gpu on edge devices
    use_gpu()
    # Get the command-line arguments
    cid = sys.argv[1]

    fleet_log(INFO,'download current model from server')
    
    with SSHClient(hostname=global_configs.VM_IP, private_key_path=global_configs.VM_KEY_PATH) as ssh:
        ssh.download_file(global_configs.SERVER_MAIN_PATH + "tmp/agg.npz", "agg.npz")
    parameters = list(np.load("agg.npz",allow_pickle = True)['arr_0'])
    os.remove("agg.npz")
    model = net_instance(f"{cid}")
    set_parameters(model, parameters)

    fleet_log(INFO,'current model downloaded')

    fleet_log(INFO,f"load the data partition for client cid {cid}")
    
    with SSHClient(hostname=global_configs.VM_IP, private_key_path=global_configs.VM_KEY_PATH) as ssh:
        ssh.download_file(global_configs.SERVER_MAIN_PATH + "tmp/partitions.npz", "partitions.npz")
    partition = np.load("partitions.npz")[cid]
    
    fleet_log(INFO,'data partition loaded')
    
    fleet_log(INFO,'load dataset to ram')
    
    trainloader, valloader, testloader = load_datasets(partition)
    
    fleet_log(INFO,'train the model')
    
    losses, accs, val_losses, val_accs = train(model, trainloader, valloader, global_configs.NUM_LOCAL_EPOCHS, plot = False,client_cid = cid)
    
    fleet_log(INFO,f"loss: {losses}, accs: {accs}, val_losses: {val_losses}, val_accs: {val_accs}")
    
    # save the model
    params = get_parameters(model)
    params = np.array(params, dtype=object)
    np.savez("tmp/res"+cid+".npz", params)
    
    fleet_log(INFO,'upload model to server')
    with SSHClient(hostname=global_configs.VM_IP, private_key_path=global_configs.VM_KEY_PATH) as ssh:
        ssh.upload_file("tmp/res"+cid+".npz", global_configs.SERVER_MAIN_PATH + "tmp/res"+cid+".npz")

    #remove the model
    os.remove("tmp/res"+cid+".npz")
    fleet_log(INFO,'done')


if __name__ == "__main__":
    main()
