from pprint import pp
from ..datasets.sc import SingleCellDataset
from .mlflow_runner import MLflowModelRunner
import argparse

parser = argparse.ArgumentParser(description='Run the model with specified parameters.')
parser.add_argument('--model-url', type=str, required=True, help='URL of the model to run')
parser.add_argument('--model-runtime-type', type=str, required=True, choices=['mlflow', 'bento', 'docker', 'mlserver'], help='Type of runtime to use for the model')
parser.add_argument('--input-data-url', type=str, required=True, help='URL of the input data')

if __name__ == '__main__':
    args = parser.parse_args()

    model_url = args.model_url
    input_data_url = args.input_data_url

    dataset = SingleCellDataset(path=input_data_url, organism='HUMAN')

    if args.model_runtime_type == 'mlflow':
        dataset = MLflowModelRunner(model_url=model_url).run(dataset)
    elif args.model_runtime_type == 'bento':
        raise NotImplementedError("Bento runtime is not implemented yet.")
    elif args.model_runtime_type == 'docker':
        raise NotImplementedError("Docker runtime is not implemented yet.")
    elif args.model_runtime_type == 'mlserver':
        raise NotImplementedError("MLServer runtime is not implemented yet.")
    else:
        raise ValueError(f"Unsupported model runtime type: {args.model_runtime_type}")

    pp(dataset.output_embedding, depth=2, compact=True, width=80)