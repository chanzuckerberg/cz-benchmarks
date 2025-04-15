FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

# Install TranscriptFormer using build secrets for authentication
RUN --mount=type=secret,id=github_token --mount=type=secret,id=github_username \
    git config --global url."https://$(cat /run/secrets/github_username):$(cat /run/secrets/github_token)@github.com/".insteadOf "https://github.com/" && \
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