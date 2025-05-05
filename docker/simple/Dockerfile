FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md

RUN pip install -e /app/package

COPY docker/simple/model.py .
COPY docker/simple/config.yaml .
COPY docker/simple/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-u", "/app/model.py"]