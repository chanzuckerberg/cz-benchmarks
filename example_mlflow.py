from czibench.datasets.types import DataType
from czibench.runners.mlflow_runner import MLflowModelRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask
from czibench.datasets.utils import init_dataset
import scanpy as sc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run MLflow model on dataset")
    parser.add_argument('--model-mode', choices=['local', 'remote'], default='local', help='Mode to run the model (default: local)')
    args = parser.parse_args()

    if args.model_mode == 'local':
        dataset = init_dataset("example-local", config_path="custom.yaml")
        runner = MLflowModelRunner(model_resource_url="model-serving/runtimes/mlflow/scvi/runtime")
    else:
        dataset = init_dataset("example-remote", config_path="custom.yaml")
        runner = MLflowModelRunner(model_endpoint="https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints/scvi4/invocations")

    dataset = runner.run(dataset)
    dataset.load_data()
    print(dataset.get_output(DataType.EMBEDDING))

    task = ClusteringTask(label_key="cell_type")
    dataset, clustering_results = task.run(dataset)
    print(clustering_results)

    task = EmbeddingTask(label_key="cell_type")
    dataset, embedding_results = task.run(dataset)
    print(embedding_results)

    sc.tl.umap(dataset.adata)
    sc.pl.umap(dataset.adata, color="cell_type")

    dataset.unload_data()
    dataset.serialize("data.dill")

if __name__ == "__main__":
    main()
