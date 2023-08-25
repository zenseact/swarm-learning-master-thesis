FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get upgrade -y && apt-get install libgl1 -y

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install flwr==1.4.0 matplotlib==3.7.1 numpy==1.23.5 opencv_python==4.7.0.72 pytest==7.3.1 pytorch_lightning==2.0.5 ray==2.2.0 torch==1.12.1 tqdm==4.65.0 zod[cli]==0.3.3
RUN pip3 install torchvision

WORKDIR /root/Fleet/fleet-learning

COPY ./ ./

RUN pip3 install -e .