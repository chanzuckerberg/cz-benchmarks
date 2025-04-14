FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

# Install TranscriptFormer
RUN git clone https://github.com/czi-ai/transcriptformer.git && \
    cd transcriptformer && \
    pip install -e .

COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md

RUN pip install -e /app/package

COPY docker/transcriptformer/model.py .
COPY docker/transcriptformer/utils.py .
COPY docker/transcriptformer/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"] 