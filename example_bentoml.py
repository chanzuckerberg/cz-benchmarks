from czibench.datasets.types import DataType
from czibench.runners.bentoml_runner import BentomlModelRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask
from czibench.datasets.utils import init_dataset
import scanpy as sc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark on Bentoml wrapped model")
    parser.add_argument('--model-mode', choices=['local', 'remote'], default='local', help='Mode to run the model (default: local)')
    args = parser.parse_args()

    if args.model_mode == 'local':
        dataset = init_dataset("example-local", config_path="custom.yaml")
        # Make sure `bentoml list` has at least one bento to serve.
        # Otherwise build a bento with `bentoml build`.
        runner = BentomlModelRunner(model_resource_url="scvi_service:latest")
    else:
        dataset = init_dataset("example-remote", config_path="custom.yaml")
        # TODO: Change the url to be the real remote url when it is ready
        runner = BentomlModelRunner(model_endpoint="http://localhost:3000")

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
