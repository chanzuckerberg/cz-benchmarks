FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

RUN git clone --depth 1 https://github.com/czi-ai/scGenePT.git && \
    (cd scGenePT && \
     git fetch --depth=1 origin 5847c57e4cccd9949a7f0b5f3775d5dcae5a5eac && \
     git checkout 5847c57e4cccd9949a7f0b5f3775d5dcae5a5eac) && \ 
    mv scGenePT/* . && rm -rf scGenePT && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install scgpt==0.2.4

# Install the czbenchmarks package
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN pip install -e /app/package[interactive] && \
    pip install cell-gears==0.0.2

COPY docker/scgenept/model.py .
COPY docker/scgenept/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"]