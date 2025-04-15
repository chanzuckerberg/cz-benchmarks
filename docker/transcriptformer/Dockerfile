FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

# Install TranscriptFormer using build args for authentication
ARG GITHUB_USERNAME
ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/" && \
    git clone https://github.com/czi-ai/transcriptformer-public transcriptformer && \
    cd transcriptformer-public && \
    pip install -e .

COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md

RUN pip install -e /app/package

COPY docker/transcriptformer/model.py .
COPY docker/transcriptformer/utils.py .
COPY docker/transcriptformer/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"] 