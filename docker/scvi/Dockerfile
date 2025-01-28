FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY docker/scvi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/scvi/model.py .
COPY docker/scvi/utils.py .
COPY docker/scvi/config.yaml .
COPY docker/scvi/hvg_names_homo_sapiens.csv.gz .
COPY docker/scvi/hvg_names_mus_musculus.csv.gz .

COPY src /app/package/src
COPY setup.py /app/package/setup.py
COPY requirements.txt /app/package/requirements.txt
COPY README.md /app/package/README.md

RUN pip install -e /app/package

ENV INPUT_PATH=/app/input/data.dill
ENV OUTPUT_PATH=/app/output/data.dill
ENV ARTIFACTS_PATH=/mnt/efs/fs1

ENTRYPOINT ["python3", "-u", "/app/model.py"]