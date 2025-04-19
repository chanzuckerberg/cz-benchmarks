# Add a Custom Model

This guide provides a step-by-step process to integrate a new model into CZ Benchmarks.

---

## Overview

To add a new model, you will:
1. Create a directory for your model.
2. Implement the necessary files and classes.
3. Extend the base classes for validation and implementation.
4. Test and integrate your model.

---

## Step 1: Create a Directory for Your Model

1. Navigate to the `docker/` directory in your project.
2. Create a new subdirectory for your model, e.g., `docker/your_model/`.
3. Structure the directory as follows:

    ```
    docker/your_model/
    ├── Dockerfile          # Define the container environment
    ├── model.py            # Implementation of your model inference code
    ├── requirements.txt    # (Optional) List Python dependencies
    ├── config.yaml         # (Optional) Configuration file for your model
    └── assets/             # (Optional) Store model weights, vocabularies, etc.
    ```

---

## Step 2: Implement the Model Validator

1. Create a validator class that extends `BaseModelValidator` or a specific validator like `BaseSingleCellValidator`.
2. Define the required properties and methods for validation.

    Example:

    ```python
    from czbenchmarks.models.validators.base_single_cell_model_validator import BaseSingleCellValidator
    from czbenchmarks.models.data_types import DataType

    class YourModelValidator(BaseSingleCellValidator):
         available_organisms = ["HUMAN", "MOUSE"]  # Use appropriate Organism enums
         required_obs_keys = []
         required_var_keys = ["feature_name"]
         model_type = "YOUR_MODEL_TYPE"

         @property
         def inputs(self):
              return {DataType.ANNDATA}

         @property
         def outputs(self):
              return {DataType.EMBEDDING}
    ```

---

## Step 3: Implement the Model Class

1. Create a model class that extends both the validator and `BaseModelImplementation`.
2. Implement the required methods, such as `run_model`.

    Example:

    ```python
    import argparse
    from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation

    class YourModel(YourModelValidator, BaseModelImplementation):
         def parse_args(self):
              parser = argparse.ArgumentParser(description="Run YourModel on input dataset.")
              parser.add_argument("--your_param", type=int, default=32, help="Description of your_param")
              return parser.parse_args()

         def get_model_weights_subdir(self, dataset):
              return "your_model"

         def _download_model_weights(self, dataset):
              # Implement your model weight download or verification logic here.
              pass

         def run_model(self, dataset):
              # Implement inference logic:
              embeddings = ...  # Run inference to produce embeddings
              dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)

    if __name__ == "__main__":
         YourModel().run()
    ```

---

## Step 4: Test Your Model

1. Ensure all dependencies are listed in `requirements.txt`.
2. Build and test the Docker container using the `Dockerfile`.
3. Validate the model's functionality by running it on a sample dataset.

---

## Additional Notes

- Refer to existing implementations like SCVI or SCGPT for inspiration.
- Use the `assets/` directory to store any additional files required by your model, such as pre-trained weights or vocabularies.
