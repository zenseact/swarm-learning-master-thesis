# Quick installation for command line tools including pytorch and ZOD sdk

* Download Anaconda Python from https://www.anaconda.com/distribution/ if you don't have it locally.

* After installation, run the command `conda init`

* Close all terminal windows currently open (the conda init step above will not have any effect on existing terminal windows - only on new ones).

* Run the following command in the terminal if you intend to use cpu (Mac or Linux)

`conda env create -f conda-environment-files/conda-environment-cpu-unix.yml`

* OR the following if you use GPU (customize cuda version if neaded). This works on Mac and Linux.

`conda env create -f conda-environment-files/conda-environment-gpu-unix.yml`

* Run

`conda activate zen`

* Then open jupyter

`jupyter notebook`

Now you have a virtual environment from which you can use all functionalities in the ZOD sdk and you have pytorch installed also!

