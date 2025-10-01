# Debugging Guide

This guide provides solutions to common issues you might encounter when using `cz-benchmarks`. Problems are organized by the stage at which they typically occur, from loading data to running tasks.


## Dataset Loading and Validation Errors

These errors usually happen when calling `load_dataset()`, `dataset.load_data()`, or `dataset.validate()`.

### 📄 File Not Found or Path Issues

  - **Error**: `FileNotFoundError` or `ValueError: Dataset path does not exist`.

    - **Cause**: The `path` specified in your YAML configuration file (e.g., `datasets.yaml`) is incorrect, or the file is not accessible.
    - **Solution**:
        1.  Verify that the path in your YAML config points to the correct `.h5ad` file.
        2.  Ensure the file exists and that you have the necessary read permissions.
        3.  If using a custom config with `load_dataset(config_path=...)`, make sure the path to the YAML file itself is correct.

### ⚙️ Dataset Not Found in Configuration

  - **Error**: `ValueError: Dataset {dataset_name} not found in config`.
    - **Cause**: The dataset name you passed to `load_dataset()` does not have a corresponding entry in the Hydra configuration.
    - **Solution**:
        1.  Check for typos in the dataset name. You can see available datasets with `list_available_datasets()`.
        2.  Ensure your custom YAML config is correctly structured and is being loaded.

### 🔬 AnnData Validation Errors

These errors originate from the content of your `.h5ad` file not meeting the requirements of the `Dataset` class.

  - **Error**: `ValueError: Dataset does not contain valid gene names...`

    - **Cause**: The `SingleCellDataset._validate()` method failed because gene names in `adata.var_names` do not start with the required prefix for the specified `organism` (e.g., `"ENSG"` for `HUMAN`).

    - **Solution**: Ensure your `AnnData` object's gene identifiers are correct. The framework can automatically use the `ensembl_id` column if it exists in `adata.var`. Check that `adata.var['ensembl_id']` contains the correct, prefixed gene IDs.

  - **Error**: `ValueError: Dataset does not contain '{key}' column in obs.`

    - **Cause**: A required metadata column is missing from `adata.obs`. This often happens with:

        - `SingleCellLabeledDataset`: The `label_column_key` (e.g., `"cell_type"`) is missing.
        - `SingleCellPerturbationDataset`: The `condition_key` is missing.

    - **Solution**: Add the required column with the correct data to your `AnnData` object's `.obs` DataFrame and save the `.h5ad` file again.

  - **Error**: `ValueError: Unexpected condition label: ``{condition}`` not present in control mapping.`

    - **Cause**: The `condition` columns in a `SingleCellPerturbationDataset` contain incorrectly formatted values.

    - **Solution**:

        - The `condition` column must follow the format ``{control_name}``, ``{control_name}_{perturbation}``, or ``{perturbation}``. Correct the values in your `adata.obs` and re-save the dataset.



## Task Execution Errors

These errors occur when calling `task.run()` and are often related to mismatches between the model output (`cell_representation`) and the dataset inputs.

### ↔️ Input Shape and Type Mismatches

  - **Error**: `ValueError: This task requires a list of cell representations` or `ValueError: This task requires a single cell representation`.

    - **Cause**: You are passing the wrong input structure to a task.

    - **Solution**:

        - For tasks with `requires_multiple_datasets = True` (like `CrossSpeciesIntegrationTask`), `cell_representation` must be a list of embeddings (`[emb1, emb2, ...]`).
        - For all other tasks, `cell_representation` must be a single `numpy.ndarray` or `pd.DataFrame`.

  - **Error**: An error related to mismatched array/DataFrame dimensions during filtering, concatenation, or model fitting (e.g., inside `filter_minimum_class` or `sklearn`).

    - **Cause**: The number of cells (rows) in your `cell_representation` does not match the number of cells in the loaded `dataset`'s metadata (e.g., `dataset.labels` or `dataset.adata.obs`).

    - **Solution**: Ensure your model's output embedding has the same number of observations and is in the same order as the cells in the original dataset file loaded by `cz-benchmarks`.

### 🎯 Task-Specific Errors

  - **Task**: `MetadataLabelPredictionTask`

  - **Issue**: Very few classes are used for training, or an error occurs in `filter_minimum_class`.

    - **Cause**: Many of your label classes have fewer samples than `min_class_size` (default is 10).

    - **Solution**: Check the distribution of your labels. If necessary, either use a dataset with more balanced classes or adjust the `min_class_size` parameter in the `MetadataLabelPredictionTaskInput`.

  - **Task**: `CrossSpeciesIntegrationTask`

  - **Error**: `AssertionError: At least two organisms are required...`

    - **Cause**: The task is being run on datasets that are all from the same species.

    - **Solution**: This task is specifically for evaluating cross-species alignment and requires inputs from at least two different organisms.


## Environment and Dependency Issues

### 📦 Missing Packages

  - **Error**: `ImportError: No module named '...'` (e.g., `hnswlib`, `scanpy`).
    - **Cause**: A required dependency is not installed in your Python environment.
    - **Solution**: Install the missing package. For all development dependencies, run `pip install -e ".[dev]"` from the repository root.

### 🧠 Memory Errors

  - **Error**: `MemoryError` or your process is killed by the OS.
    - **Cause**: Loading or processing data (e.g., a large `.h5ad` file or a dense embedding matrix) requires more RAM than is available.
    - **Solution**:
        1.  Run your code on a machine with more RAM.
        2.  If possible, for initial debugging, create a smaller version of your dataset by subsampling cells.
        3.  Check for parts of your code that might be making unnecessary copies of large objects.



## General Tips for Debugging

- **Start Small**: When debugging, use a small, fast-running dataset (like `tsv2_bladder`) or a subset of your custom data to quickly iterate.
- **Isolate the Problem**: Determine if the issue is in data loading or task execution. First, ensure your dataset loads and validates successfully:
    ```python
    dataset = load_dataset("my_dataset", config_path="my_config.yaml")
    dataset.load_data()
    dataset.validate()
    print(dataset.adata)
    ```
- **Check Shapes and Types**: Before running a task, print the shapes and types of your inputs to catch mismatches early.
    ```python
    print(f"Embedding shape: {my_embedding.shape}")
    print(f"Labels length: {len(dataset.labels)}")
    print(f"Embedding type: {type(my_embedding)}")
    ```

- **Use a Debugger**: Use an interactive debugger like `pdb` or your IDE's built-in debugger to step through the code, inspect variables, and understand the execution flow.
    ```python
    import pdb; pdb.set_trace()
    ```
- **Dependency Conflicts**: Ensure all dependencies are installed in a clean virtual environment. Recreate the environment if needed.

- **hnswlib package installation error**: If the `hnswlib` package fails to install with an error like `fatal error: Python.h: No such file or directory`, ensure you have installed Python development headers files and static libraries. On Ubuntu, this can be done via `sudo apt-get install python3-dev`.