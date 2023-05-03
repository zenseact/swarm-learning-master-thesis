import paramiko
import time
from paramiko import RSAKey
from common.logger import log
from logging import INFO

def run(ip, cid):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    private_key = RSAKey.from_private_key_file('/root/.ssh/id_rsa')
    ssh.connect(ip, username='nvidia', pkey=private_key)

    repo_location = '/home/nvidia/Fleet/fleet-learning'
    
    channel = ssh.invoke_shell()
    channel.send(f'cd {repo_location} && git pull \n')
    time.sleep(5)
    channel.send(f'cd {repo_location} && nohup python3 edge_main.py {cid} > output.log 2>&1 &\n')
    time.sleep(5)
    # Read the output of the 'echo $!' command
    channel.send("echo $?; exit\n")
    output = channel.recv(1024).decode('utf-8')
    nohup_pid = output.split('\n')

    log(INFO, f"Nohup process output for ID: {nohup_pid}")
    channel.close()

    ssh.close()