# Create an **MLflow Model** package
1. Make sure you are in `model_pkgs/mlflow_pkgs/transcriptformer_mlflow_pkg/` directory:

```
$ cd model_pkgs/mlflow_pkgs/transcriptformer_mlflow_pkg
```

2. Create a `uv` virtual environment and install `mlflow` and `transcriptformer`:

```
$ uv venv --python=3.11
$ source .venv/bin/activate
$ uv pip install mlflow
$ uv pip install transcriptformer 
```

3. Download model weights and auxiliary data to the `model_data` directory.

```
$ transcriptformer download tf-sapiens --checkpoint-dir model_data/
```

4. Run `transcriptformer_mlflow_packager.py` to create the **MLflow Model** artifact.

```
$ python transcriptformer_mlflow_packager.py --model-variant tf_sapiens   --checkpoint-path model_data/tf_sapiens --output-dir mlflow_models --requirements transcriptformer_requirements.txt
```

5. This should have created a directory `mlflow_models/transcriptformer_tf_sapiens/`. `transcriptformer_tf_sapiens` is the **MLflow Model** artifact and it should have the following structure:

```
├── MLmodel
├── artifacts
│   └── tf_sapiens
│       ├── config.json
│       ├── model_weights.pt
│       └── vocabs
│           ├── assay_vocab.json
│           └── homo_sapiens_gene.h5
├── code
│   └── model_code
│       ├── __init__.py
│       └── transcriptformer_mlflow_model.py
├── conda.yaml
├── input_example.json
├── python_env.yaml
├── python_model.pkl
├── requirements.txt
├── serving_input_example.json
```

# Verify that model package can be used to run inference
1. Create a test input json payload from the **MLflow Model Signature**:

```
$ python generate_mlflow_test_payload.py --model-uri mlflow_models/transcriptformer_tf_sapiens --json-payload-filepath test_input_payload.json
```

2. Flesh out the values in `test_input_payload.json`. Example:

```
$ cat test_input_payload.json

{
  "dataframe_split": {
    "columns": [
      "input_file",
      "output_file"
    ],
    "data": [
      [
        "/home/ssm-user/.cz-benchmarks/datasets/example_small.h5ad",
        "tf_results/example_small_embeddings.h5ad"
      ]
    ]
  },
  "params": {
    "gene_col_name": "ensembl_id",
    "precision": "16-mixed",
    "pretrained_embedding": "",
    "batch_size": 16
  }
}
```

3. Run inference with the payload in its own `uv` virtual environment:

```
$ mlflow models predict --model-uri mlflow_models/transcriptformer_tf_sapiens --content-type json --input-path test_input_payload.json --output-path test_output.json --env-manager uv 
```

4. Verify that the inference ran correctly by checking the `test_output.json` file:

```
$ cat test_output.json
{"predictions": [{"output_file": "tf_results/example_small_embeddings.h5ad"}]}

$ python
Python 3.11.12 (main, Apr  9 2025, 04:04:00) [Clang 20.1.0 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import anndata as ad
>>> adata = ad.read_h5ad("tf_results/example_small_embeddings.h5ad")
>>> adata.obsm.keys()
KeysView(AxisArrays with keys: embeddings)
>>> adata.obsm['embeddings'][:10]
array([[-0.14298186, -0.12608503,  0.04042066, ...,  0.05823774,
         0.11938392,  0.05610898],
       [-0.24399517, -0.2638534 , -0.16482367, ...,  0.07856863,
         0.03534734, -0.00754361],
       [-0.4980944 , -0.40106925, -0.24204014, ..., -0.4581786 ,
         0.44571778,  0.17741683],
       ...,
       [-0.20021866, -0.31380177, -0.2203846 , ..., -0.0949255 ,
         0.20720059, -0.00074428],
       [-0.3758839 , -0.1579138 , -0.13913535, ..., -0.04498409,
         0.08683188,  0.04261744],
       [-0.53276694, -0.4542373 , -0.48934302, ...,  0.05328108,
         0.6291807 ,  0.02545681]], shape=(10, 2048), dtype=float32)
>>> quit()
```
