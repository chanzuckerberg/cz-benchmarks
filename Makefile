.PHONY: all scvi clean

# Default target
all: scvi scgpt

# Build the scvi image
scvi: 
	docker build -t czibench-scvi:latest -f docker/scvi/Dockerfile .

scgpt:
	docker build -t czibench-scgpt:latest -f docker/scgpt/Dockerfile .

# Clean up images
clean:
	docker rmi czibench-scvi:latest || true
	docker rmi czibench-scgpt:latest || true
# Helper target to rebuild everything from scratch
rebuild: clean all