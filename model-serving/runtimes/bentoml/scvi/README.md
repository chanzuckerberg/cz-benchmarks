## Create Bento Artifact
1. `cd model-serving/runtimes/bentoml/scvi
2. Install bento and scvi python package requirements 
```
pip install -r requirements.txt
```
3. Run setup to copy model weights and other artifacts. This should create an `artifacts` folder
```
chmod +x setup.sh
./setup.sh
```
4. Build a bento artifact
```
bentoml build
```
5. You should see the list of bentos
```
bentoml list
```
## Serve Model
1. Serve the latest build of the model server via HTTP
```
bentoml serve scvi_service:latest
```
2. Now you are ready to run inference via HTTP. Here is an example:

```
curl -X POST -H "Content-Type: application/json" \
  -d '{"file_path":"/Users/psridharan/code/cz-benchmarks/model-serving/runtimes/bentoml/scvi/example.h5ad","organism":"homo_sapiens"}' \
  http://127.0.0.1:3000/predict
```

# Serve Model In Docker Container
**NOTE**: This will NOT work for arm64 architectures. So run this on an amd64 architecture.

1. Create a docker image for any of the bentos. First run, `bentoml list` and pick a bento artifact to containerize. Example:
```
bentoml containerize scvi_service:xyn4xwxezsmjecww
```

2. Run the model server by mounting a path where input files exist. Example:
```
docker run --rm -p 3000:3000 -v /home/ssm-user/cz-benchmarks/model-serving/runtimes/bentoml/scvi:/my_data scvi_service:deenhrhe2sraocww
```

3. Now you are ready to run inference via HTTP. Here is an example:

```
curl -X POST -H "Content-Type: application/json" \
  -d '{"file_path":"/my_data/example.h5ad","organism":"homo_sapiens"}' \
  http://127.0.0.1:3000/predict
``` 

