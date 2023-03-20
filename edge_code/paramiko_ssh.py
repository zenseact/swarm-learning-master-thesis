import paramiko

class SSHClient:
    def __init__(self, hostname, private_key_path):
        self.hostname = hostname
        self.private_key_path = private_key_path
        self.ssh = None
        self.sftp = None

    def __enter__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private_key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
        self.ssh.connect(hostname=self.hostname, username='', pkey=private_key)
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
