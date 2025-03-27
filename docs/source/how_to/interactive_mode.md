# Running a Docker Container in Interactive Mode

This guide explains how launch an interactive docker container with cz-benchmarks and run benchmarks for a model.

## Command Line Flags

The script `scripts/run_docker.sh` is used to launch the container and accepts the following command line flags:

- `-m, --model-name MODEL_NAME`: (Required) Set the model name (geneformer, scgenept, scgpt, scvi, uce). This also determines which docker image to use.
- `-h, --help`: Display usage information and exit.

Example:
```bash
bash scripts/run_docker.sh -m MODEL_NAME
```

## User Defined Configuration

The script also includes user-configurable variables at the top of the file:

### Mount Paths
- `DATASETS_CACHE_PATH`: Path to the cache directory containing datasets. Default: `${HOME}/.cz-benchmarks/datasets`
- `MODEL_WEIGHTS_CACHE_PATH`: Path to the cache directory for model weights. Default: `${HOME}/.cz-benchmarks/weights`
- `DEVELOPMENT_CODE_PATH`: Path to the workstation development code. Default: `$(pwd)`. Set to blank or remove if mounting code is not desired.

### Container Execution Settings
- `EVAL_CMD`: Command to execute when the container starts. Default: `bash`
- `RUN_AS_ROOT`: Whether to run the container as root. Options: `false` (default) or `true`

## Running the Interactive Example

The repository includes an example script (`examples/example_interactive.py`) that demonstrates how to run various tasks on a dataset with the user-selected model. To use this example:

1. Launch the container with the appropriate model:
   ```bash
   bash scripts/run_docker.sh -m MODEL_NAME
   ```

2. Inside the container, run the example script:
   ```bash
   python3 examples/example_interactive.py
   ```

The example script will:

- Download the specified dataset(s)
- Evaluate the dataset(s) with the model
- Execute tasks
- Print the results of each task

## Important Notes

- When running in development mode (with `DEVELOPMENT_CODE_PATH` set), the code directory is mounted
- The container has GPU support enabled by default and is configured with appropriate memory settings
- The script validates that all required directories exist before starting the container and that a valid model is provided
- The script will automatically ensure the appropriate container is downloaded and current

## Limitations

- Each model is provided in a separate Docker container, thus limiting this method to a single model per container. 
- Running multiple models can be accomplished by launching separate containers for each model. The imported model name must be updated:
```python
from model import SCVI as BenchmarkModel
```