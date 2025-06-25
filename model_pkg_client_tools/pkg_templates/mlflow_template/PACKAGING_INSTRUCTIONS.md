# Why use MLflow to package models
**FILL-ME-IN**

# Setup
1. Create and activate a python virtual environment. We recommend `uv`, `conda` or `mamba` but you can use any virtual environment library you like. We also recommend `python>=3.11`.

2. Install `uv` in the virtual environment. Example:

```
$ pip install --upgrade uv 
```

**NOTES:**

- We use `uv` to install packages because its dependency resolution
is much faster and more robust than `pip`.

- This step is not necessary if you already have `uv` installed on your system.

3. Install `mlflow` in the virtual environment. Example:

```
$ uv pip install mlflow==3.1.0
```

4. Generate the directory structure in which you will implement the `mlflow` wrapper for your model:

```
$ python generate_model_pkg_dir_structure.py --model-name <model-name> --target <target-directory> --pkg-template mlflow_template --model-archetype file_uri_to_tensor
```

As an example, you should see a directory structure that looks like this after running `generate_model_pkg_dir_structure.py`:

```
$ tree transcriptformer_mlflow_pkg

transcriptformer_mlflow_pkg
├── mlflow_packager.py
├── model_code
│   ├── __init__.py
│   └── transcriptformer_mlflow_model.py
├── model_data
├── model_spec.py
└── requirements.in
```
5. Fill out the `requirements.in` file inside the generated directory. Example:

```
$ cat requirements.in

# Provide your dependencies below
transcriptformer>=0.3.0
```

6. Generate a `requirements.txt` file from `requirements.in`. This will capture all the transitive dependencies of the packages you listed in `requirements.in`:

```
$ uv pip compile requirements.in -o requirements.txt
```

7. Download all necessary artifacts (ex: model weights and auxiliary data) to the `model_data` directory. Example:

```
$ transcriptformer download tf-sapiens --checkpoint-dir model_data/
```

# Complete the implementation of the **MLflow PythonModel** wrapper
1. Complete the implementation of the following methods in the `model_code/<model-name>_mlflow_model.py` file:

```
def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
    raise NotImplementedError

def _get_input(self, uri: str, **params) -> Any:
    raise NotImplementedError

def _forward(self, input_obj: Any, **params) -> np.ndarray:
    raise NotImplementedError
```

# Create an **MLflow Model** package
1. Run `mlflow_packager.py` to create the **MLflow Model** artifact. See the thorough module docstring for `mlflow_packager.py` or get thorough CLI usage documentation by typing `python mlflow_packager.py --help`.

Example Usage:

```bash
python mlflow_packager.py \
    --model-class model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel \
    --checkpoint model_data/tf_sapiens \
    --model-config-json '{"model_variant":"tf_sapiens"}' \
    --model-tag model_variant=tf_sapiens \
    --example-uri tiny.h5ad \
    --output-shape '[-1,2048]' \
    --output-dtype float32 \
    --extra-params-spec-json '{"gene_col_name":"string", "precision":"string", "pretrained_embedding":"string", "batch_size":"integer"}' \
    --out-dir mlflow_model_artifacts 
```


3. Check that it created a directory that has this structure:

```
├── MLmodel
├── artifacts
├── code
│   └── model_code
│       ├── __init__.py
│       └── [model-name]_mlflow_model.py
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
$ python predict_func.py --model-uri <mlflow-model-directory> --json-payload-file test_input_payload.json
```
4. **Run inference to test if the model can be served from a REST API endpoint.**_This tests if the json input payload round-trips through mlflow's native support for deserialization, datatype validation and finally serialization of the output back to json._
The inference process will spin up its own virtual environment.
This likely means you will use the same virtual environment library (conda, uv, virtualenv) recommended by the model's installation instructions. See [documentation](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-models-predict).

```
$ mlflow models predict --model-uri <mlflow-model-directory> --content-type json --input-path test_input_payload.json --output-path test_output.json --env-manager <virtualenv-manager>
```

5. Verify that the inference ran correctly by checking the `test_output.json` file.
