from czibench.datasets.types import DataType
from czibench.runners.mlflow_runner import MLflowModelRunner
from czibench.runners.bentoml_runner import BentomlModelRunner
from czibench.runners.sagemaker_mlflow_runner import SageMakerMLflowRunner
from czibench.runners.sagemaker_runner import SageMakerRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask
from czibench.datasets.utils import init_dataset
import scanpy as sc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run scvi model on dataset")
    parser.add_argument('--model-runtime', choices=['mlflow', 'sagemaker', 'sagemaker-mlflow', 'bentoml'], default='mlflow', help='Model runtime type (default: mlflow)')
    parser.add_argument('--model-mode', choices=['local', 'remote'], default='local', help='Mode to run the model (default: local)')
    args = parser.parse_args()

    if args.model_mode == 'local':
        if args.model_runtime == 'sagemaker':
            dataset = init_dataset("example-local-sagemaker", config_path="custom.yaml")
            runner = SageMakerRunner(model_resource_url="file://model-serving/runtimes/sagemaker/scvi/scvi_model_code.tar.gz")
        elif args.model_runtime == 'mlflow':
            dataset = init_dataset("example-local-mlflow", config_path="custom.yaml")
            runner = MLflowModelRunner(model_resource_url="model-serving/runtimes/mlflow/scvi/runtime")
        elif args.model_runtime == 'bentoml':
            runner = BentomlModelRunner(model_resource_url="scvi_service:latest")
    else: # remote
        if args.model_runtime == 'sagemaker':
            dataset = init_dataset("example-remote-sagemaker", config_path="custom.yaml")
            runner = SageMakerRunner(model_endpoint="scvi-mlflow")
        if args.model_runtime == 'sagemaker-mlflow':
            dataset = init_dataset("example-remote-sagemaker", config_path="custom.yaml")
            runner = SageMakerMLflowRunner(model_endpoint="scvi-mlflow-20250314140011")
        if args.model_runtime == 'mlflow':
            dataset = init_dataset("example-remote-mlflow", config_path="custom.yaml")
            runner = MLflowModelRunner(model_endpoint="https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints/scvi5/invocations")
    
    if runner is None:
        raise ValueError(f"Unsupported model runtime type: {args.model_runtime} ({args.model_mode})")
        
    # runner = ContainerRunner(
    #     image="czibench-scvi:latest",
    #     gpu=True,
    # )

    # Run local or remote inference
    # Note: For both local and remote inference, only dataset.path is needed!
    dataset = runner.run(dataset)
    
    # Load 
    dataset.load_data()
    print(f"Shape of embedding attached to benchmarking dataset: {dataset.get_output(DataType.EMBEDDING).shape}")
    print(f"Embedding attached to benchmarking dataset: {dataset.get_output(DataType.EMBEDDING)}")
    
    print(f"Running ClusteringTask...")
    task = ClusteringTask(label_key="cell_type")
    dataset, clustering_results = task.run(dataset)
    print(f"ClusteringTask results: {clustering_results}")

    print(f"Running EmbeddingTask...")
    task = EmbeddingTask(label_key="cell_type")
    dataset, embedding_results = task.run(dataset)
    print(f"EmbeddingTask results: {embedding_results}")
    
    print("About to display umap...")
    sc.tl.umap(dataset.adata)
    sc.pl.umap(dataset.adata, color="cell_type")
    print("Finished displaying umap...")
    
    print("Unloading benchmarking dataset from memory and serializing to disk...")
    dataset.unload_data()
    dataset.serialize("data.dill")
    print("Finished unloading benchmarking dataset and serializing to disk")

if __name__ == "__main__":
    main()
