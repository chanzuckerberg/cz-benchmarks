FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Geneformer and its dependencies
RUN git clone https://huggingface.co/ctheodoris/Geneformer && \
    cd Geneformer && \
    pip install .

# Copy local files
COPY docker/geneformer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/geneformer/model.py .
COPY docker/geneformer/config.yaml .
COPY docker/geneformer/gene_dictionaries /app/gene_dictionaries

# Install the czibench package
COPY src /app/package/src
COPY setup.py /app/package/setup.py
COPY requirements.txt /app/package/requirements.txt
COPY README.md /app/package/README.md
RUN pip install -e /app/package

# Set environment variables for data paths
ENV INPUT_PATH=/app/input/data.dill
ENV OUTPUT_PATH=/app/output/data.dill
ENV ARTIFACTS_PATH=/mnt/efs

# Add memory management environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=1

ENTRYPOINT ["python3", "-u", "/app/model.py"]