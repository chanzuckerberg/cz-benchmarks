FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential git wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    apt-get install -y --no-install-recommends nvidia-cuda-toolkit && \
    pip install --upgrade pip && \
    pip install virtualenv && \
    virtualenv /opt/venv_aido -p python3.10 && \
    /opt/venv_aido/bin/pip install --upgrade pip 


# Install CUDA Toolkit 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda

# Clone the ModelGenerator repository so that model references resolve.
RUN git clone https://github.com/genbio-ai/ModelGenerator.git && \
    mv ModelGenerator/* /app && rm -rf ModelGenerator


   
# Install the modelgenerator package and its dependencies.
# NOTE: Installing from AIDO repo is causing issues as the changes are causing problem running AIDO.Cell
# /opt/venv_aido/bin/pip install ".[flash_attn]" && \
# Install the dependencies for flash attention for AIDO.Cell manually
RUN /opt/venv_aido/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/venv_aido/bin/pip install  flash_attn==2.5.6  && \
    /opt/venv_aido/bin/pip install /app && \
    /opt/venv_aido/bin/pip install -r /app/experiments/AIDO.Cell/requirements.txt


COPY docker/aido/cell_utils.py /app/cell_utils.py
COPY docker/aido/OS_scRNA_gene_index.19264.tsv /app/OS_scRNA_gene_index.19264.tsv
COPY docker/aido/model.py /app/model.py

# Install the cz-benchmarks package.
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN /opt/venv_aido/bin/pip install -e /app/package

ENV PYTHONPATH="/app/package/src:$PYTHONPATH"
# # Specify entrypoint so that the model is executed when the container starts.
ENTRYPOINT ["/opt/venv_aido/bin/python3", "-u", "/app/model.py"]


