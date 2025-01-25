.PHONY: all scvi uce clean

# Default target
all: scvi uce scgpt

# Build the scvi image
scvi: 
	docker build -t czibench-scvi:latest -f docker/scvi/Dockerfile .

uce:
	docker build -t czibench-uce:latest -f docker/uce/Dockerfile .

scgpt:
	docker build -t czibench-scgpt:latest -f docker/scgpt/Dockerfile .

# Clean up images
clean:
	docker rmi czibench-scvi:latest || true
	docker rmi czibench-uce:latest || true
	docker rmi czibench-scgpt:latest || true
  
# Helper target to rebuild everything from scratch
rebuild: clean all