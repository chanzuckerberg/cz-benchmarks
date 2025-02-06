This repo is self-container project with its own Python virtual environment requirements. 
It can be added to the parent cz-benchmark project VS Code workspace using "File" > "Add Folder to Workspace".

Setup:
```
for ORGANISM in homo_sapiens mus_musculus; do
    mkdir -p artifacts/${ORGANISM}
    cp ../../docker/scvi/hvg_names_${ORGANISM}.csv.gz artifacts/${ORGANISM}/hvg_names.csv.gz
    aws s3 --no-sign-request cp s3://cellxgene-contrib-public/models/scvi/2024-07-01/${ORGANISM}/model.pt artifacts/${ORGANISM}
done
aws --profile virtual-cells-dev s3 cp s3://generate-cross-species/datasets/test/example.h5ad .
```

To package and run the model via Python, use the configured Run tasks (`.vscode/launch.json`):
* "Package Model": Packages an MLflow model, saving in ./runtime directory (git ignored)
* "Run model (mlflow)": Runs the packaged ML model model (from ./runtime) in batch mode.
* "Run model (debug)": Runs the model directly from the `mlflow_scvi.py` module (not using the packaged MLflow model), allowing for debugging from within VS Code (e.g. breakpointing).

To run the model as batch process:
```
mlflow models predict --env-manager uv -m models/scvi/runtime <<EOF
{"inputs": [["models/scvi/example.h5ad"]]}
EOF
```

To run the model as local API server:
```
# Serve the model
mlflow models serve -m runtime --env-manager uv

# Send a request to the model
curl http://localhost:5000/invocations -H "Content-Type:application/json"  --data '{"inputs": [["example.h5ad"]]}'> example-output.json
```

# Databricks Usage

To log models to Databricks and to calls MLflow models served on Databricks, you will need obtain a access token. See https://docs.databricks.com/en/mlflow/access-hosted-tracking-server.html#access-the-mlflow-tracking-server-from-outside-databricks

1. Install Databricks CLI (https://docs.databricks.com/en/dev-tools/cli/install.html#install)
```
mlflow brew tap databricks/tap
brew install databricks
```
2. Create and set an access token:
```
databricks api post /api/2.0/token/create | jq .token_value | databricks configure --host https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com token
```

To perform inference on a Databricks model endpoint, where the model accepts an S3 object URL as input:
```
curl \
  -u token:<DATABRICKS_TOKEN> \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[<S3_OBJECT_URL>]]}' \
  https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints/<MODEL_SERVING_ENDPOINT_NAME>/invocations`
```
You must specify the `DATABRICKS_TOKEN`, `S3_OBJECT_URL`, and `MODEL_MODEL_SERVING_ENDPOINT_NAME` values. The S#





