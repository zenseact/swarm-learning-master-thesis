import numpy as np
from edge_code.paramiko_ssh import SSHClient
import os
import sys
from edge_code.data_loader import load_datasets
from common.utilities import train, use_gpu

if __name__ == '__main__':
    # use gpu on edge devices
    use_gpu()
    # Get the command-line arguments
    cid = sys.argv[1]
    strategy = sys.argv[2]

    # download current model from server
    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.download_file("/root/Fleet/tmp/agg.npz", "agg.npz")
    model = np.load("agg.npz")
    os.remove("agg.npz")

    # load the data
    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.download_file("/root/Fleet/tmp/partitions.npz", "partitions.npz")
    partition = np.load("partitions.npz")['0'][cid]
    data = load_datasets(partition)

    # train the model
    # FILL FUNCTION
    model = train(model, data)

    # save the model
    np.savez("res"+cid+".npz",model)

    with SSHClient(hostname='172.25.16.67', private_key_path='/home/nvidia/.ssh/id_rsa') as ssh:
        ssh.upload_file("res"+cid+".npz", "/root/Fleet/edge_com/res"+cid+".npz")

    #remove the model
    os.remove("res"+cid+".npz")