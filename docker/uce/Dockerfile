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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to path and create Python 3.11 environment
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda create -n uce python=3.11 -y && \
    conda init bash && \
    echo "conda activate uce" >> ~/.bashrc

# Clone the UCE repository and install UCE
RUN git clone https://github.com/giovp/UCE && \
    cd UCE && \
    /opt/conda/envs/uce/bin/pip install . && \
    cd ..

# Install the czbenchmarks package
COPY src /app/package/src
COPY conf /app/package/conf
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN /opt/conda/envs/uce/bin/pip install -e /app/package

# Create model_files directory and empty CSV file for species protein embeddings
RUN mkdir -p /app/model_files && \
    echo "species,path" > /app/model_files/new_species_protein_embeddings.csv

# Install base requirements
COPY docker/uce/requirements.txt .
RUN /opt/conda/envs/uce/bin/pip install -r requirements.txt

# Copy model files
COPY docker/uce/uce_model.py .
COPY docker/uce/config.yaml .

ENTRYPOINT ["/opt/conda/envs/uce/bin/python", "-u", "/app/uce_model.py"]
