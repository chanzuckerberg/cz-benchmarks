FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY docker/scgpt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/scgpt/model.py .
COPY docker/scgpt/config.yaml .

# Install the cz_benchmarks package
COPY src /app/package/src
COPY setup.py /app/package/setup.py
COPY requirements.txt /app/package/requirements.txt
COPY README.md /app/package/README.md
RUN pip install -e /app/package

# Set environment variables for data paths
ENV INPUT_PATH=/app/input/data.dill
ENV OUTPUT_PATH=/app/output/data.dill
ENV ARTIFACTS_PATH=/mnt/efs

ENTRYPOINT ["python3", "-u", "/app/model.py"]