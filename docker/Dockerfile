FROM continuumio/miniconda3

# Site package update
RUN apt-get update && apt-get upgrade -y && apt-get install libgl1 -y

WORKDIR /root/Fleet/fleet-learning

COPY ./ ./

# Conda environment  (fleet)
RUN conda env create -f conda-environment-files/conda-environment-cpu-unix.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "fleet", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fleet", "python3", "<insert_script.py>"]