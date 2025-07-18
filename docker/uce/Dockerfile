FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
# TODO: remove conda as it does not appear to be needed
RUN wget -O miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash miniforge3.sh -b -p /opt/conda && \
    rm miniforge3.sh

# Add conda to path and create Python 3.11 environment
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda create -n uce python=3.11 -y && \
    conda init bash && \
    echo "conda activate uce" >> ~/.bashrc

# Clone the UCE repository and install UCE
# TODO: clone / install somewhere that won't be squashed by code mounting
RUN git clone --depth 1 https://github.com/giovp/UCE && \
    (cd UCE && \
     git fetch --depth=1 origin dc5421593920b821f789a21a7aae308ac237f908 && \
     git checkout dc5421593920b821f789a21a7aae308ac237f908 && \
    /opt/conda/envs/uce/bin/pip install .)

# Install the czbenchmarks package
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
COPY README-pypi.md /app/package/README-pypi.md
RUN /opt/conda/envs/uce/bin/pip install -e /app/package[interactive]

# Create model_files directory and empty CSV file for species protein embeddings
RUN mkdir -p /app/model_files && \
    echo "species,path" > /app/model_files/new_species_protein_embeddings.csv

# Create protein embeddings directory
RUN ln -s /weights/model_files/protein_embeddings /app/model_files/protein_embeddings

# Install base requirements
# Note: this overrides some package versions specified by the UCE repo that were previously installed.
COPY docker/uce/requirements.txt .
RUN /opt/conda/envs/uce/bin/pip install -r requirements.txt

# Copy model files
# TODO: change filename to model.py
COPY docker/uce/uce_model.py .
COPY docker/uce/config.yaml .

ENTRYPOINT ["/opt/conda/envs/uce/bin/python", "-u", "/app/uce_model.py"]
