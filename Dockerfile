FROM nvidia/cuda:latest

# Install system dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        openssh-server \
        unzip \
        screen \
        vim \
        net-tools \
 && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME /opt/miniconda
ENV PATH ${MINICONDA_HOME}/bin:${PATH}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x Miniconda3-latest-Linux-x86_64.sh \
 && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
 && rm Miniconda3-latest-Linux-x86_64.sh
RUN conda update -n base -c defaults conda
# RUN conda install python=3.6

# sshd Fix
RUN mkdir /var/run/sshd

# JupyterLab
RUN conda install -c conda-forge jupyterlab ipywidgets nodejs'>10.0.0'

# Project
COPY . /pytorch-vq-vae
WORKDIR /pytorch-vq-vae

# Start container in notebook mode
CMD SHELL=/bin/bash jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root

# docker build -t pytorch-vq-vae .
# docker run -v /host/directory/data:/data --runtime=nvidia --network=host --ipc=host --gpus all -it pytorch-vq-vae
