FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11

# Clone the UCE repository and install UCE into the uce virtual environment
RUN git clone https://github.com/giovp/UCE && \
    cd UCE && \
    pip install . && \
    cd ..

# Install base requirements
COPY docker/uce/requirements.txt .
RUN pip install -r requirements.txt

# Copy model files
COPY docker/uce/uce_model.py .
COPY docker/uce/config.yaml .
COPY docker/uce/model_files/new_species_protein_embeddings.csv model_files/new_species_protein_embeddings.csv

# Install the czibench package
COPY src /app/package/src
COPY setup.py /app/package/setup.py
COPY requirements.txt /app/package/requirements.txt
COPY README.md /app/package/README.md
RUN pip install -e /app/package

# Set environment variables for data paths
ENV INPUT_PATH=/app/input/data.dill
ENV OUTPUT_PATH=/app/output/data.dill

ENTRYPOINT ["python3", "/app/uce_model.py"]
