# How to Run Benchmarks


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