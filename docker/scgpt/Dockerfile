FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY docker/scgpt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the czbenchmarks package
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
COPY README-pypi.md /app/package/README-pypi.md
RUN pip install -e /app/package[interactive]

COPY docker/scgpt/model.py .
COPY docker/scgpt/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"]