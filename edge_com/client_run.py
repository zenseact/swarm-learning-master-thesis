import paramiko
import time
from paramiko import RSAKey
from pygit2 import Repository

def run(ip, cid):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    private_key = RSAKey.from_private_key_file('/root/.ssh/id_rsa')
    ssh.connect(ip, username='nvidia', pkey=private_key)

    repo_location = '/home/nvidia/Fleet/fleet-learning'
    
    branch = Repository('.').head.shorthand

    channel = ssh.invoke_shell()
    channel.send(f'cd {repo_location} && git checkout {branch} \n')
    time.sleep(3)
    channel.send(f'cd {repo_location} && git fetch --all \n')
    time.sleep(5)
    channel.send(f'cd {repo_location} && git reset --hard origin/{branch} \n')
    time.sleep(5)
    channel.send(f'cd {repo_location} && nohup python3 edge_main.py {cid} > output.log 2>&1 &\n')
    time.sleep(5)
    channel.close()

    ssh.close()