# Interactive Debugging Guide

## Overview

This guide explains how to use interactive debugging mode with cz-benchmarks containers. Interactive mode allows you to:

- Access a bash shell inside the container
- Mount your local development directory
- Debug code in real-time
- Make changes to the codebase without rebuilding

## Build model inference container

In the project root run `make <model>` to build the docker image ex: `make geneformer`
## Starting Interactive Mode

```python
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference

# Load your dataset
dataset = load_dataset("example", config_path="~/cz-benchmarks/example.yaml")

# Run inference in interactive mode
dataset = run_inference(
    model_name="GENEFORMER",  # or any other model
    dataset=dataset,
    interactive=True,
    app_mount_dir="/home/ssm-user/cz-benchmarks"  # Your local development directory
)
```

## Accessing the Container

1. Get the container ID:

```shell
docker ps
```

2. Attach to the container:

```shell
docker attach <container_id>
```

## Development Workflow

### 1. Mounting Your Development Directory

- The `app_mount_dir` parameter mounts your local directory to `/app` in the container
- This allows you to make changes to your local files and have them reflected in the container
- Example: `/home/ssm-user/cz-benchmarks` → `/app` in container

### 2. Installing in Development Mode


```shell
# Inside the container
cd /app
pip install -e .
```

This installs cz-benchmarks in editable mode, so changes to your local files are immediately reflected.

### 3. Adding Debugger Statements

Add Python debugger statements in your code:

```python
import pdb; pdb.set_trace()  # Code will pause here when executed
```

### 4. Running the Model

```shell
# Inside the container
cd /app/docker/<model>/  # e.g., /app/docker/geneformer/
python model.py
```

⚠️ **Important**: Run `model.py` from the mounted directory (`/app/docker/<model>/`), not from `/app/`. The version in `/app/` was copied during the Docker build and won't reflect your changes.

## Cleanup

1. Remove all debugger statements from your code
2. Rebuild the container if you're done with interactive mode:

```shell
make geneformer # or whatever model you're working with, run make command from project root
```

## Tips

- Use `Ctrl+P` then `Ctrl+Q` to detach from the container without stopping it
- The data directory is mounted at `/raw` in the container
- Changes to your local files are immediately reflected after installing in editable mode
- Keep track of which files you've modified to ensure they're properly cleaned up

## Troubleshooting

### Common Issues

1. **Changes not reflecting**: Make sure you've installed the package in editable mode
2. **Wrong file location**: Ensure you're editing files in the mounted directory, not the container's original files
3. **Debugger not working**: Verify you're running the correct version of `model.py` from the mounted directory

### Getting Help

If you encounter issues:

1. Check the container logs: `docker logs <container_id>`
2. Verify your mounts: `docker inspect <container_id>`
3. Ensure you're in the correct directory when running the model