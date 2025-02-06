from pprint import pp
from ..datasets.sc import SingleCellDataset
from .mlflow_runner import MLflowModelRunner
import argparse

parser = argparse.ArgumentParser(description='Run the model with specified parameters.')
parser.add_argument('--model-resource-url', type=str, required=False, help='URL of the model to run')
parser.add_argument('--model-endpoint', type=str, required=False, help='Endpoint of the model to run')
parser.add_argument('--model-runtime-type', type=str, required=True, choices=['mlflow', 'bento', 'docker', 'mlserver'], help='Type of runtime to use for the model')
parser.add_argument('--input-data-url', type=str, required=True, help='URL of the input data')

def call_mlflow_model(dataset, model_url):
    print(f"Calling MLflow model at {model_url} with dataset {dataset}")
    

if __name__ == '__main__':
    args = parser.parse_args()

    model_resource_url = args.model_resource_url
    input_data_url = args.input_data_url

    # TODO: Dataset objects do not support remote URL paths for data, so this is not currently a proper design for working with remotely-stored input data.
    dataset = SingleCellDataset(path=input_data_url, organism='HUMAN')

    if args.model_runtime_type == 'mlflow':
        dataset = MLflowModelRunner(model_resource_url=model_resource_url, model_endpoint=args.model_endpoint).run(dataset)
    elif args.model_runtime_type == 'bento':
        raise NotImplementedError("Bento model execution is not implemented yet.")
    elif args.model_runtime_type == 'docker':
        raise NotImplementedError("Docker model execution is not implemented yet.")
    elif args.model_runtime_type == 'mlserver':
        raise NotImplementedError("MLServer model execution is not implemented yet.")
    else:
        raise ValueError(f"Unsupported model runtime type: {args.model_runtime_type}")

    pp(dataset.output_embedding, depth=2, compact=True, width=80)