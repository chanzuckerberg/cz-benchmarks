FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Geneformer and its dependencies
RUN git clone https://huggingface.co/ctheodoris/Geneformer && \
    cd Geneformer && \
    git checkout 69e6887 && \
    pip install .

COPY docker/geneformer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the czbenchmarks package
COPY src /app/package/src
COPY conf /app/package/conf
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN pip install -e /app/package

COPY docker/geneformer/model.py .
COPY docker/geneformer/config.yaml .

# Add memory management environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=1

ENTRYPOINT ["python3", "-u", "/app/model.py"]