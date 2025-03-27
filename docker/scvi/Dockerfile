FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY docker/scvi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/package/src
COPY conf /app/package/conf
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md

RUN pip install -e /app/package

COPY docker/scvi/model.py .
COPY docker/scvi/utils.py .
COPY docker/scvi/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"]