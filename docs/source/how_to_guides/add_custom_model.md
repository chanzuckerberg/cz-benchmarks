# Add a Custom Model

This guide provides a step-by-step process to integrate your own model into CZ Benchmarks.

---

## Overview

To add a new model, you will:
1. Create a directory for your model.
2. Implement the necessary files and classes, extending the base classes for model implementation and input validation.
3. Test and integrate your model.

---

## Step 1: Create a Directory for Your Model

1. Navigate to the `docker/` directory in your project.
2. Create a new subdirectory for your model, e.g., `docker/your_model/`.
3. Structure the directory as follows:

    ```
    docker/your_model/
    ├── Dockerfile          # Define the container environment
    ├── model.py            # Implementation of your model inference code
    ├── config.yaml         # Configuration file for your model
    ├── requirements.txt    # (Optional) List Python dependencies
    └── assets/             # (Optional) Store model weights, vocabularies, etc.
    ```

---

## Step 2: Add a new ModelType

In the `src.czbenchmarks.models.types.ModelType` enum, add a value for your model:

     Example:

     ```
     class ModelType(Enum):
          ...
          YOUR_MODEL = "YOUR_MODEL"
     ```


## Step 3: Implement the Model Class

1. Create a model class that extends `BaseModelImplementation`.
2. Implement the required methods, such as `run_model`.

     Example:

     ```python
     import argparse
     from typing import Set
     from czbenchmarks.datasets.types import DataType
     from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
     from czbenchmarks.models.types import ModelType


     class YourModel(BaseModelImplementation):
          def parse_args(self):
               parser = argparse.ArgumentParser(description="Run YourModel on input dataset.")
               parser.add_argument("--your_param", type=int, default=32, help="Description of your_param")
               return parser.parse_args()

          model_type = ModelType.YOUR_MODEL

          @property
          def inputs(self) -> Set[DataType]:
                # Specify appropriate `DataType`s below
               return { }

          @property
          def outputs(self) -> Set[DataType]:
               # Specify appropriate `DataType`s below (embeddings are a typical model output)
               return { DataType.EMBEDDING }  

          def get_model_weights_subdir(self, dataset) -> str:
               return "your_model"

          def _download_model_weights(self, dataset) -> None:
               # Implement your model weight download or verification logic here.
               pass

          def run_model(self, dataset):
               # Implement inference logic:
               embeddings = ...  # Run inference to produce embeddings
               dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)

     if __name__ == "__main__":
           YourModel().run()
     ```

Note that you can access any arguments specified in the `config.yaml` or via command-line options using `self.args`.

---

## Step 3: (Optional) Extend `BaseSingleCellValidator`

If your model is a single-cell transcriptomic model and accepts AnnData objects as input, then it can extend `BaseSingleCellValidator`. This will enable the class to validate that the input `Dataset` provides the required organisms, obs keys, and var keys.

1. Add `BaseSingleCellValidator` as a parent class.
2. Specify the required organisms, obs keys, and var keys that are defined as class variables.
3. Specify `DataType.ANNDATA` as the model's input type via the `inputs()` method.
    Example:

     ```python
     ...
     from czbenchmarks.datasets.types import Organism
     from czbenchmarks.models.validators.base_single_cell_model_validator import BaseSingleCellValidator


     class YourModel(BaseModelImplementation, BaseSingleCellValidator):

          ...

          available_organisms = [Organism.HUMAN, Organism.MOUSE]  # Use appropriate Organism enums
          required_obs_keys = []  # Specify required obs keys, as needed
          required_var_keys = ["feature_name"]  # Use appropriate feature name

          @property
          def inputs(self) -> Set[DataType]:
               return { DataType.ANNDATA }

          ...
     ```

---

## Step 4: Create a Config File for the Model

1. Create a `config.yaml` file in your model's directory. This file will define the configuration parameters required for your model.
2. Include the `_target_` key to specify the model class and any additional parameters your model requires.

     Example:

     ```yaml
     _target_: model.YourModel
     your_param: 32
     another_param: "value"
     ```

    The config file may include any additional parameters required by your model.

---

## Step 5: Add `requirements.txt`

1. Create a `requirements.txt` under `docker/your_model`.
2. Add required Python packages

## Step 6: Create a `Dockerfile`

1. Create a new file `docker/your_model/Dockerfile`
2. Specify Docker commands to build the Docker image, per the requirments of the model.

     Example:
     ```
     FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

     WORKDIR /app

     RUN apt-get update && \
     apt-get install -y python3 python3-pip

     COPY docker/your_model/requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     COPY src /app/package/src
     COPY pyproject.toml /app/package/pyproject.toml
     COPY README.md /app/package/README.md

     RUN pip install -e /app/package[interactive]

     COPY docker/your_model/model.py .
     COPY docker/your_model/config.yaml .
     # Specify additional files here, as neeeded

     ENTRYPOINT ["python3", "-u", "/app/model.py"]
     ``` 

## Step 5: Build and Test Your Model

1. Build the Docker container using the `Dockerfile` you created. Run the following command, replacing `your_model` with the appropriate values:

     ```sh
     docker build -t cz-benchmarks-models:your_model -f docker/your_model/Dockerfile .
     ```

3. Add the Docker build command to your project's `Makefile` for easier execution. For example:

     ```makefile
     .PHONY: your_model
     your_model:
          docker build -t cz-benchmarks-models:your_model -f docker/your_model/Dockerfile .
     ```

4. Test the Docker container to ensure it works as expected. You can run the container and verify its functionality by executing your model on a sample dataset.
5. Verify that your model works as expected by testing it on a sample dataset. Ensure the outputs are correct and align with the intended task and metric.

---

## Additional Notes

- For guidance, review existing implementations such as `docker/scvi` or `docker/scgpt`. These examples can help you understand best practices and common patterns.
- Use the `assets/` directory to store supplementary files your model might need, such as pre-trained weights, vocabularies, or other resources. Keeping these files organized ensures your model remains portable and easy to manage.

