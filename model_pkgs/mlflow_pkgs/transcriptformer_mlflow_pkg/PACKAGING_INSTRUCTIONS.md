# Why use MLflow to package models
**FILL-ME-IN**

# Create an **MLflow PythonModel** inference interface 
1. Confirm that you are in a directory structure that looks like this:

```
transcriptformer_mlflow_pkg
        ├── PACKAGING_INSTRUCTIONS.md
        ├── generate_mlflow_test_payload.py
        ├── model_code
        │   ├── __init__.py
        │   └── transcriptformer_mlflow_model.py
        ├── model_data
        ├── mlflow_packager.py
        └── transcriptformer_requirements.txt
``` 
2. Flesh out the package dependencies required to run your model in `transcriptformer_requirements.txt`.

3. Install `transcriptformer_requirements.txt` in whatever virtual environment is recommended by the installation instructions in the `transcriptformer` repo.

4. Download model weights and auxiliary data to the `model_data` directory.

5. Flesh out the implementation of `model_code/transcriptformer_mlflow_model.py`.

# Create an **MLflow Model** package
1. Flesh out the implementation of `mlflow_packager.py`.

2. Run `mlflow_packager.py` to create the **MLflow Model** artifact.

3. Check that it created a directory that has this structure:

```
├── MLmodel
├── artifacts
├── code
│   └── model_code
│       ├── __init__.py
│       └── transcriptformer_mlflow_model.py
├── conda.yaml
├── input_example.json
├── python_env.yaml
├── python_model.pkl
├── requirements.txt
```

# Verify that model package can be used to run inference
1. Create a test input json payload from the **MLflow Model Signature**:

```
$ python generate_mlflow_test_payload.py --model-uri <mlflow-model-directory> --json-payload-filepath test_input_payload.json
```

2. Flesh out the values in `test_input_payload.json`.

3. **Run inference** to test that a **python function invocation of inference succeeds and datatype validation passes** so that it can be included in a python script or jupyter notebook.

```
$ python test_predict_fn.py --model-uri <mlflow-model-directory> --json-payload-file test_input_payload.json
```
4. **Run inference to test if the model can be served from a REST API endpoint.**_This tests if the json input payload round-trips through mlflow's native support for deserialization, datatype validation and finally serialization of the output back to json._
The inference process will spin up its own virtual environment.
This likely means you will use the same virtual environment library (conda, uv, virtualenv) recommended by the model's installation instructions. See [documentation](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-models-predict).

```
$ mlflow models predict --model-uri <mlflow-model-directory> --content-type json --input-path test_input_payload.json --output-path test_output.json --env-manager <virtualenv-manager>
```

5. Verify that the inference ran correctly by checking the `test_output.json` file.
