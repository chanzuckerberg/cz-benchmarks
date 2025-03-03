.PHONY: start predict stop

CURL=curl -s
MODEL_NAME=scvi
MLFLOW_MODEL_RUNTIME=../../runtimes/mlflow/$(MODEL_NAME)/runtime
INPUT_FILE=../../../example_small.h5ad

start:
	@if [ ! -d "$(MLFLOW_MODEL_RUNTIME)" ]; then \
		echo "Missing packaged MLflow model at $(MLFLOW_MODEL_RUNTIME)."; \
		exit 1; \
	fi
	mlserver start .

metrics:
	$(CURL) http://localhost:8082/metrics

list:
	$(CURL) -H 'content-type: application/json' http://localhost:8080/v2/repository/index -d '{}' | jq .

model_metadata:
	$(CURL) http://localhost:8080/v2/models/$(MODEL_NAME) | jq .

unload:
	$(CURL) -X POST http://localhost:8080/v2/repository/models/$(MODEL_NAME)/unload | jq .

predict:
	@OUTPUT_FILE=$(shell mktemp) && \
	$(CURL) -H 'content-type: application/json' http://localhost:8080/v2/models/$(MODEL_NAME)/infer \
	-d '{"inputs": [{ "name": "input-0", "datatype": "BYTES", "shape": [-1, 1], "data": ["$(INPUT_FILE)"]}]}' \
	-o $$OUTPUT_FILE && \
	jq . $$OUTPUT_FILE | head -n 30

stop:
	pkill -f 'mlserver stop'