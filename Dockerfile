FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install dependencies
COPY apt_install.txt .
RUN apt-get update && apt-get install -y `cat apt_install.txt`

# Config pip
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip, install py libs
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt --upgrade

RUN pip3 install jax[cuda11_cudnn805] -f  https://storage.googleapis.com/jax-releases/jax_releases.html --upgrade --force-reinstall
RUN pip3 install -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==0.1.73+cuda11.cudnn805
RUN pip3 install --upgrade --force-reinstall git+https://github.com/deepmind/dm-haiku

RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY . .
