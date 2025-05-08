FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y software-properties-common build-essential ninja-build git wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-pip virtualenv libopenblas-dev libomp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    virtualenv /opt/venv_aido -p python3.10 && \
    /opt/venv_aido/bin/pip install --upgrade pip setuptools wheel packaging hatchling && \
    /opt/venv_aido/bin/pip install --no-cache-dir --no-build-isolation torch==2.6.0 torchvision==0.21.0 && \
    /opt/venv_aido/bin/pip install flash_attn==2.5.6 --no-build-isolation
 
RUN git clone https://github.com/genbio-ai/ModelGenerator.git /app/ModelGenerator && \
    cd /app/ModelGenerator && \
    git checkout db0777d9d8b677ce452536a048f156768ac81ba7 && \
    mv * /app && \
    cd /app && \
    rm -rf /app/ModelGenerator

RUN /opt/venv_aido/bin/pip install /app 

COPY docker/aido/model.py /app/model.py
COPY docker/aido/config.yaml /app/config.yaml

# Install the cz-benchmarks package.
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN /opt/venv_aido/bin/pip install -e /app/package[interactive]

ENV PYTHONPATH="/app/package/src:$PYTHONPATH"
# Specify entrypoint so that the model is executed when the container starts.
ENTRYPOINT ["/opt/venv_aido/bin/python3", "-u", "/app/model.py"]
