# Quick Start Guide

Welcome to **cz-benchmarks**! This guide will help you get started with installation, setup, and running your first benchmark in just a few steps.

## Requirements

Before you begin, ensure you have the following installed:

- 🐍 **[Python 3.10+](https://www.python.org/downloads/)**:  3.10+**: Ensure you have Python 3.10 or later installed.
- 🐳 **[Docker](https://docs.docker.com/get-started/get-docker/)**: Required for container-based execution.
- 💻 **Hardware**: Intel/AMD64 architecture CPU with NVIDIA GPU, running Linux with [NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html).


## Installation

You can install the library using one of the following methods:

### Option 1: Install from [PyPI](https://pypi.org/project/cz-benchmarks/) (Recommended)

The easiest way to install the library is via PyPI:

```bash
pip install cz-benchmarks
```

### Option 2: Install from Source (For Development)

If you plan to contribute or debug the library, install it from source:

1. Clone the repository:

    ```bash
    git clone https://github.com/chanzuckerberg/cz-benchmarks.git
    cd cz-benchmarks
    ```

2. Install the package:

    ```bash
    pip install .
    ```

3. For development, install in editable mode with development dependencies:

    ```bash
    pip install -e ".[dev]"
    ```

## Running Benchmarks

You can run benchmarks using the CLI or programmatically in Python.

### 💻 Using the CLI

The CLI simplifies running benchmarks. Below are common commands:

#### 🔍 List Available Benchmark Assets

```bash
czbenchmarks list models
czbenchmarks list datasets
czbenchmarks list tasks
```

#### 🏃 Run a Benchmark

```bash
czbenchmarks run \
  --models SCVI \
  --datasets tsv2_bladder \
  --tasks clustering \
  --label-key cell_type \
  --output-file results.json
```

#### 🔧 CLI Run Options

Below are the key options available for running benchmarks via the CLI:

- **`--models`**: Specifies the model to use (e.g., `SCVI`).

- **`--datasets`**: Specifies the dataset to benchmark (e.g., `tsv2_bladder`).

- **`--tasks`**: Defines the evaluation task(s) to execute (e.g., `clustering`).

- **`--label-key`**: The metadata key to use as labels for the task (e.g., `cell_type`).

- **`--output-file`**: File path to save the benchmark results (e.g., `results.json`).

> 💡 **Tip**: Combine these options to customize your benchmark runs effectively.

> 📁 **Output**: Results will be saved to `results.json`.

#### 📖 Get Help

Use the `--help` flag to explore available commands and options:

```bash
czbenchmarks --help
czbenchmarks <command> --help
```

### 🐍 Using the Python API

The library can also be used programmatically. Here's an example:

```python
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import ClusteringTask

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

## Next Steps

Explore the following resources to deepen your understanding:
- **How-to Guides**: [Practical guides](./how_to_guides/index.rst) for using and extending the library.
- **Setup Guides**: [Setup Guides](./how_to_guides/setup_guides.md)
- **Developer Docs**: [Internal structure and extension points](./developer_guides/index.rst).
- **GitHub Repository**: [cz-benchmarks](https://github.com/chanzuckerberg/cz-benchmarks) for troubleshooting and support.

Happy benchmarking! 🚀
