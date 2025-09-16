# Quick Start Guide

Welcome to **cz-benchmarks**! This guide will help you get started with installation, setup, and running your first benchmark in just a few steps.

## Requirements

Before you begin, ensure you have the following installed:

- 🐍 **[Python 3.10+](https://www.python.org/downloads/)**: Ensure you have Python 3.10 or later installed.
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

## Command-Line Interface (CLI)

Use the **cz-benchmarks** CLI to list supported datasets and tasks.

- **List all available datasets**:
    ```bash
    czbenchmarks list datasets
    ```

- **List all available tasks**:
    ```bash
    czbenchmarks list tasks
    ```

For a full list of options, run:
```bash
czbenchmarks list --help
```

## Running Benchmarks

The cz-benchmarks package is designed to be used programmatically within your Python workflow. The below code example demonstrates how you can run a benchmark task on a model's output. The example generates a "dummy" cell embedding as the model's output and computes a benchmark result using the clustering task.

> Note that if you are interested in running benchmarks using a CLI instead of code, you can use the [Virtual Cell Platform CLI](https://chanzuckerberg.github.io/vcp-cli/). The VCP CLI supports:
> * Running a benchmark on any Virtual Cell Platform [model](https://virtualcellmodels.cziscience.com/models) that has been [benchmarked](https://virtualcellmodels.cziscience.com/benchmarks) using a `cz-benchmark` dataset, allowing you to reproduce a VCP-published result.
> * Running a benchmark on any Virtual Cell Platform [model](https://virtualcellmodels.cziscience.com/models) that has been [benchmarked](https://virtualcellmodels.cziscience.com/benchmarks) using your own benchmarking dataset.
> * Running a benchmark on the output of your own model using either a `cz-benchmark` dataset or your own benchmarking dataset.


```python
import numpy as np
from czbenchmarks.datasets import load_dataset
from czbenchmarks.tasks import ClusteringTask, ClusteringTaskInput

# 1. Load a benchmark dataset
# This dataset has pre-defined labels we can use for evaluation.
dataset = load_dataset("tsv2_bladder")

# 2. Generate or load your model's cell embedding
# For this example, we'll generate a **dummy embedding**.
# In a real scenario, this would be the output of your ML Model.
n_obs = dataset.adata.n_obs
n_features = 128
my_model_embedding = np.random.rand(n_obs, n_features)

# 3. Instantiate the desired evaluation task
# We'll use the ClusteringTask to see how well the embedding
# separates known cell types.
clustering_task = ClusteringTask()

# 4. Prepare the input for the task
# The task needs the ground-truth labels from the dataset.
task_input = ClusteringTaskInput(
    obs=dataset.adata.obs,
    input_labels=dataset.labels
)

# 5. Run the task and get the results
# The task evaluates your embedding against the ground-truth labels.
results = clustering_task.run(
    cell_representation=my_model_embedding,
    task_input=task_input
)

# 6. Print the results
# The output will contain metrics like Adjusted Rand Index (ARI)
# and Normalized Mutual Information (NMI).
print(results)
```

## Next Steps

Explore the following resources to deepen your understanding:
- **How-to Guides**: [Practical guides](./how_to_guides/index.rst) for using and extending the library.
- **Setup Guides**: [Setup Guides](./how_to_guides/setup_guides.md)
- **Developer Docs**: [Internal structure and extension points](./developer_guides/index.rst).
- **GitHub Repository**: [cz-benchmarks](https://github.com/chanzuckerberg/cz-benchmarks) for troubleshooting and support.

Happy benchmarking! 🚀
