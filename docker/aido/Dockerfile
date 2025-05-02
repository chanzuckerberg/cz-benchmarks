FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential git wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip virtualenv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    virtualenv /opt/venv_aido -p python3.10 && \
    /opt/venv_aido/bin/pip install --upgrade pip setuptools wheel packaging && \
    /opt/venv_aido/bin/pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    /opt/venv_aido/bin/pip install flash_attn==2.7.4.post1

# Install required Python packages
RUN /opt/venv_aido/bin/pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    /opt/venv_aido/bin/pip install flash_attn==2.7.4.post1

# Clone the ModelGenerator repository
RUN git clone https://github.com/genbio-ai/ModelGenerator.git && \
    mv ModelGenerator/* /app && rm -rf ModelGenerator

# Install the modelgenerator package and its dependencies.
RUN /opt/venv_aido/bin/pip install /app && \
    /opt/venv_aido/bin/pip install -r /app/experiments/AIDO.Cell/requirements.txt


# COPY docker/aido/cell_utils.py /app/cell_utils.py
COPY docker/aido/model.py /app/model.py

# Install the cz-benchmarks package.
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN /opt/venv_aido/bin/pip install -e /app/package

ENV PYTHONPATH="/app/package/src:$PYTHONPATH"
# # Specify entrypoint so that the model is executed when the container starts.
ENTRYPOINT ["/opt/venv_aido/bin/python3", "-u", "/app/model.py"]


