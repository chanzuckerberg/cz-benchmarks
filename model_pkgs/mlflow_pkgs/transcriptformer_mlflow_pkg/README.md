# Create an **MLflow Model** package
1. Make sure you are in `model_pkgs/mlflow_pkgs/transcriptformer_mlflow_pkg/` directory:

```
$ cd model_pkgs/mlflow_pkgs/transcriptformer_mlflow_pkg
```

2. Create a `uv` virtual environment and install `mlflow` and `transcriptformer`:

```
$ uv venv --python=3.11
$ source .venv/bin/activate
$ uv pip install -r transcriptformer_requirements.txt 
```

3. Download model weights and auxiliary data to the `model_data` directory.

```
$ transcriptformer download tf-sapiens --checkpoint-dir model_data/
```

4. Run `mlflow_packager.py` to create the **MLflow Model** artifact.

```
$ python mlflow_packager.py --model-variant tf_sapiens   --checkpoint-path model_data/tf_sapiens --output-dir mlflow_models --requirements transcriptformer_requirements.txt
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
      "input_uri"
    ],
    "data": [
      [
        "/home/ssm-user/.cz-benchmarks/datasets/example_small.h5ad"
      ]
    ]
  },
  "params": {
    "gene_col_name": "ensembl_id",
    "precision": "16-mixed",
    "pretrained_embedding": "",
    "batch_size": -1
  }
}
```

3. **Run inference** to test that a **python function invocation of inference succeeds and datatype validation passes** so that it can be included in a python script or jupyter notebook.

```
$ python predict_func.py --model-uri mlflow_models/transcriptformer_tf_sapiens --json-payload-file test_input_payload.json
```
4. **Run inference to test if the model can be served from a REST API endpoint.**_This tests if the json input payload round-trips through mlflow's native support for deserialization, datatype validation and finally serialization of the output back to json._ Run inference with the payload in its own `uv` virtual environment:

```
$ mlflow models predict --model-uri mlflow_models/transcriptformer_tf_sapiens --content-type json --input-path test_input_payload.json --output-path test_output.json --env-manager uv 
```

5. Verify that the inference ran correctly by checking the `test_output.json` file:

```
$ head -c 1000 test_output.json | less
{"predictions": [[[-0.1429818570613861, -0.12608502805233002, 0.040420662611722946, -0.19157767295837402, 0.22647874057292938, 0.15660904347896576, -0.13574904203414917, 0.0721682608127594, 0.024216821417212486, -0.045611537992954254, -0.3007250130176544, 0.08299852907657623, 0.000504956697113812, 0.003926896024495363, 0.08237996697425842, 0.18843598663806915, -0.008095023222267628, -0.012726053595542908, -0.11273445188999176, 0.03558430075645447, -0.05046337470412254, 0.18308386206626892, 0.07628823816776276, 0.017358627170324326, 0.027970347553491592, -0.33074551820755005, 0.07161068916320801, 0.15132226049900055, -0.06798345595598221, 0.06222779303789139, -0.13658343255519867, 0.2187119573354721, 0.1522756665945053, 0.023600874468684196, -0.1128731444478035, 0.0877125933766365, 0.045885197818279266, 0.05724436417222023, 0.16931083798408508, -0.1518353968858719, 0.03722844272851944, -0.13497602939605713, -0.015310755930840969, -0.10012456774711609, -0.032233309000730515, 0.2051643580
```