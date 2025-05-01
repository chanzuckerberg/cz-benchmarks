# Add a Custom Model

This guide provides a step-by-step process to integrate your own model into CZ Benchmarks.

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
    ├── config.yaml         # Configuration file for your model
    ├── requirements.txt    # (Optional) List Python dependencies
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

3. Create config file for the model

    Example:

    ```yaml
     _target_: model.YourModel

    ```
    Config file may include additional parameters required by model

---

## Step 4: Test Your Model

1. Ensure all required dependencies for your model are listed in the `requirements.txt` file. This ensures the Docker container has everything it needs to run your model.

2. Build the Docker container using the `Dockerfile` you created. Run the following command, replacing `your_model_name` and `your_model` with the appropriate values:

     ```sh
     docker build -t cz-benchmarks-models:your_model_name -f docker/yourmodelpath/Dockerfile .
     ```

3. Add the Docker build command to your project's `Makefile` for easier execution. For example:

     ```makefile
     .PHONY: your_model_name
     your_model_name:
          docker build -t cz-benchmarks-models:your_model_name -f docker/yourmodelpath/Dockerfile .
     ```

4. Test the Docker container to ensure it works as expected. You can run the container and verify its functionality by executing your model on a sample dataset.
5. Verify that your model works as expected by testing it on a sample dataset. Ensure the outputs are correct and align with the intended task and metric.

---

## Additional Notes

- For guidance, review existing implementations such as `scVI` or `scGPT`. These examples can help you understand best practices and common patterns.
- Use the `assets/` directory to store supplementary files your model might need, such as pre-trained weights, vocabularies, or other resources. Keeping these files organized ensures your model remains portable and easy to manage.

