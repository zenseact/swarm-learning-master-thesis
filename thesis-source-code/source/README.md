# Quick installation for command line tools including pytorch and ZOD sdk

* Download Anaconda Python from https://www.anaconda.com/distribution/ if you don't have it locally.

* After installation, run the command `conda init`

* Close all terminal windows currently open (the conda init step above will not have any effect on existing terminal windows - only on new ones).

* Run the following command in the terminal 

`conda env create -f conda-environment-files/conda-environment-gpu-unix.yml`

* Note! You need to change the version of CUDA drivers in the yml script before running the following command so you don't get inconsistent version with TorchVision  

* Run

`conda activate zen`

Now you have a virtual environment from which you can use all functionalities in the ZOD sdk and you have pytorch installed also!

