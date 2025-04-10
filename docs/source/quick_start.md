# Quick Start Guide

## Requirements

-   🐍 **Python 3.10+**: Ensure you have Python 3.10 or later installed.
-   🐳 **Docker**: Required for container-based execution.
-   ⚙️ **Optional**: Install `uv` for managing dependencies.

## Installation from Source

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/chanzuckerberg/cz-benchmarks)

To install the package from source, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/chanzuckerberg/cz-benchmarks.git
    cd cz-benchmarks
    ```

2. Install the package:

    ```
    pip install .
    ```

3. For development, install in editable mode with development dependencies:

    ```
    pip install -e ".[dev]"
    ```

### Using `uv` for Dependency Management

For managing dependencies with `uv`, follow these steps:

1. Install `uv` using pip or follow the [official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/) :

    ```
    pip install uv
    ```

2. Install python version

    ```
    uv python install
    ```

3. Sync all extras:

    ```
    uv sync --all-extras
    ```

⚠️ **Note**: Ensure Docker is logged out of public ECR to avoid authentication issues:

```
docker logout public.ecr.aws
```

### macOS Development Setup

For macOS users, ensure the following prerequisites are installed:

1. **Conda**: Install Conda (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)).
2. **Compiler Tools**: Install Xcode Command Line Tools:

    ```
    xcode-select --install
    ```

Once the prerequisites are installed, proceed with the setup:

1. Install `hnswlib` using Conda:

    ```
    conda install -c conda-forge hnswlib
    ```

2. Install the package in editable mode with development dependencies:

    ```
    pip install -e ".[dev]"
    ```

#### Additional Installation Notes

- It is highly recommended to create a virtual environment before installing dependencies. This ensures a clean and isolated Python environment, avoiding conflicts with system-wide packages. To create and activate a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

  Once activated, proceed with the installation steps.
- For detailed installation instructions using `uv`, refer to the [official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- Ensure all dependencies are installed before running the benchmarks.
- Docker is required for container-based execution, so verify that Docker is properly installed and configured.
- If you encounter issues, consult the [GitHub repository](https://github.com/chanzuckerberg/cz-benchmarks) for troubleshooting and support.



## Running Benchmarks

To run the benchmark with CLI `czbenchmarks`, use the following command to run a benchmark:

```
czbenchmarks run \
    --models SCVI \
    --datasets tsv2_bladder \
    --tasks clustering \
    --label-key cell_type \
    --output-file results.json
```

This command runs the `clustering` task using the `SCVI` model on the `tsv2_bladder` dataset, with results saved to `results.json`.

### CLI Usage

The `czbenchmarks` CLI provides several commands for managing and running benchmarks. Below are some common usage examples:

- **List available options**:
S
    ```
    czbenchmarks list datasets|models|tasks
    ```

- **Run a benchmark**:

    ```
    czbenchmarks run \
        --models SCVI \
        --datasets tsv2_bladder \
        --tasks clustering \
        --label-key cell_type
    ```

- **Display help information**:


    ```
    czbenchmarks --help
    ```

    or command specific help:

    ```
    czbenchmarks <command> --help
    ```

### Python Usage

The `czbenchmarks` can also be used programmatically in Python. Here's an example of how to use it:

```python
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import ClusteringTask, EmbeddingTask

# Load a dataset
dataset = load_dataset("tsv2_bladder")

# Run inference using the SCVI model
dataset = run_inference("SCVI", dataset)

# Perform clustering on the dataset
clustering = ClusteringTask(label_key="cell_type")
results = clustering.run(dataset)

# Print the clustering results
print(results)
```

This example demonstrates how to load a dataset, run inference with a model, and execute a clustering task programmatically.

#### Additional Running Benchmarks Notes

- Ensure all dependencies are installed before running the CLI or library commands.
- Refer to the [GitHub repository](https://github.com/chanzuckerberg/cz-benchmarks) for more examples and troubleshooting tips.
- For containerized execution, verify that Docker is properly installed and configured.
- Use the `--help` flag with any command to explore additional options and configurations.
- The `label_key` parameter should match the column name in your dataset that contains the labels for tasks like clustering.
- Save your results to a file (e.g., `results.json`) for further analysis or sharing.



## Supported Models

| Model      | Description                                   |
|------------|-----------------------------------------------|
| SCVI       | Variational inference for single-cell RNA-seq |
| SCGPT      | GPT for transcriptomics                       |
| Geneformer | Transformer model for gene expression         |
| scGenePT   | Gene perturbation prediction transformer      |
| UCE        | Universal Cell Embedding model                |



## Supported Evaluation Tasks

Each task implements `_run_task()` and `_compute_metrics()`

-   **ClusteringTask**: Uses Leiden clustering on embeddings
-   **EmbeddingTask**: Evaluates separation of classes
-   **LabelPredictionTask**: Cross-validation-based prediction
-   **IntegrationTask**: Batch correction and cell type preservation
-   **PerturbationTask**: MSE and correlation on predicted vs. ground truth perturbations
    

## Supported Metrics

Metrics are organized by tags:

-   `clustering`: ARI, NMI
-   `embedding`: Silhouette score
-   `integration`: Entropy per cell, Batch silhouette
-   `perturbation`: MSE, R2, Jaccard
-   `label_prediction`: Accuracy, F1, Precision, Recall, AUROC