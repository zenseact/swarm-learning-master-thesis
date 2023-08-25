# Federated Fleet Learning

## Environment setup
To execute the scripts, create an environment and install the dependencies using the following steps:

Create a new vitual environment:

```bash
python3 -m venv .venv
```

Activate the environment:
```bash
source .venv/bin/activate
```

If make is not installed, install it:
```bash
apt install make
```

Make sure you have the latest pip version:
```bash
python3 -m pip install --upgrade pip
```

Finally, install all the dependencies with 
```bash
make install-dev
```

If you are on a Nvidia edge node, make sure Cuda and Nvidia cuda compiler driver is installed:
```bash
apt show cuda
nvcc --version
```

If your device has a GPU, install cude compatible packages with
```bash
make install-gpu
```

Now you have a virtual environment!

## Mounting the ZOD dataset
If the dataset is not mounted on your node, mount it with the following steps.
This tool might be needed:

```bash
sudo apt install nfs-common
```
To find out what you can mount:
```bash
showmount -e 172.25.16.66
```
To mount the new ZOD2 dataset on a VM or Edge Device in the edge lab:
```bash
sudo mkdir /mnt/ZOD2
```
```bash
sudo chmod 775 /mnt/ZOD2
```
```bash
sudo mount 172.25.16.66:/ZOD2_clone_vlan2002 /mnt/ZOD2
```
```bash
sudo vi /etc/fstab
```

Add: 
```bash
172.25.16.66:/ZOD2_clone_vlan2002 /mnt/ZOD2 nfs defaults 0 0
```

## Ground truth generation
If the mounted dataset does not contain a ground truth, this needs to be generated. To create the target variables (ground truth) for ZOD, run the main function in common/groundtruth_utils.py. This will store a dictionary with frame id -> target in the global_configs.STORED_GROUND_TRUTH_PATH path. - NOTE: there will be many failed frames that looks like errors.