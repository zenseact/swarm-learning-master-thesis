import paramiko

class SSHClient:
    def __init__(self, hostname, username, password):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.ssh = None
        self.sftp = None

    def __enter__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.hostname, username=self.username, password=self.password)
        self.sftp = self.ssh.open_sftp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()

    def download_file(self, remote_path, local_path):
        self.sftp.get(remote_path, local_path)

    def upload_file(self, local_path, remote_path):
        self.sftp.put(local_path, remote_path)