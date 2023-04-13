import paramiko
import time
from paramiko import RSAKey

def run(ip, cid):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    private_key = RSAKey.from_private_key_file('/root/.ssh/id_rsa')
    ssh.connect(ip, username='nvidia', pkey=private_key)

    repo_location = '/home/nvidia/Fleet/fleet-learning'
    
    channel = ssh.invoke_shell()
    channel.send(f'cd {repo_location} && git pull\n')
    time.sleep(5)
    channel.send(f'cd {repo_location} && nohup python edge_main.py {cid} > output.log 2>&1 &\n')
    time.sleep(5)
    channel.close()

    ssh.close()