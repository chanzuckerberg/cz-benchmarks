.PHONY: all scvi clean

# Default target
all: scvi

# Build the scvi image
scvi: 
	docker build -t czibench-scvi:latest -f docker/scvi/Dockerfile .

# Clean up images
clean:
	docker rmi czibench-scvi:latest || true

# Helper target to rebuild everything from scratch
rebuild: clean all