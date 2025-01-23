.PHONY: all scvi uce clean

# Default target
all: scvi uce

# Build the scvi image
scvi: 
	docker build -t czibench-scvi:latest -f docker/scvi/Dockerfile .

uce:
	docker build -t czibench-uce:latest -f docker/uce/Dockerfile .

# Clean up images
clean:
	docker rmi czibench-scvi:latest || true
	docker rmi czibench-uce:latest || true

# Helper target to rebuild everything from scratch
rebuild: clean all