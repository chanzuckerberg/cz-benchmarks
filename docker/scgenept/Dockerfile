FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip git

RUN git clone https://github.com/czi-ai/scGenePT.git && \
    mv scGenePT/* . && rm -rf scGenePT && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install scgpt    

# Install the czbenchmarks package
COPY src /app/package/src
COPY pyproject.toml /app/package/pyproject.toml
COPY README.md /app/package/README.md
RUN pip install -e /app/package

COPY docker/scgenept/model.py .
COPY docker/scgenept/config.yaml .

ENTRYPOINT ["python3", "-u", "/app/model.py"]