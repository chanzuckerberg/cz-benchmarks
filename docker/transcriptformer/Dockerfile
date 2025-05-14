FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

# Clone the repository and install
RUN apt-get install -y curl && \
    pip install uv && \
    git clone --branch v0.1.0 --depth 1 https://github.com/czi-ai/transcriptformer.git && \
    cd transcriptformer && \
    uv venv --python=3.11 && \
    . .venv/bin/activate && \   
    uv pip install -e .

COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md

RUN . /app/transcriptformer/.venv/bin/activate && uv pip install -e /app/package

COPY docker/transcriptformer/model.py .

ENTRYPOINT ["/app/transcriptformer/.venv/bin/python", "-u", "/app/model.py"]