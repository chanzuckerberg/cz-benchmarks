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
6. Serve the latest build of the model server via HTTP
```
bentoml serve scvi_service:latest
```
7. Now you are ready to run inference via HTTP. Here is an example:

```
curl -X POST -H "Content-Type: application/json" \
  -d '{"file_path":"/Users/psridharan/code/cz-benchmarks/model-serving/runtimes/bentoml/scvi/example.h5ad","organism":"homo_sapiens"}' \
  http://127.0.0.1:3000/predict
```