import scanpy as sc

from cz_benchmarks.datasets.utils import load_dataset
from cz_benchmarks.runner import ContainerRunner
from cz_benchmarks.tasks.sc import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

if __name__ == "__main__":
    dataset = load_dataset("example", config_path="custom.yaml")
    runner = ContainerRunner(
        image="cz_benchmarks-scvi:latest",
        gpu=True,
    )

    dataset = runner.run(dataset)

    task = ClusteringTask(label_key="cell_type")
    dataset, clustering_results = task.run(dataset)

    task = EmbeddingTask(label_key="cell_type")
    dataset, embedding_results = task.run(dataset)

    task = MetadataLabelPredictionTask(label_key="cell_type")
    dataset, prediction_results = task.run(dataset)

    print(prediction_results)
    print(embedding_results)

    sc.tl.umap(dataset.adata)
    sc.pl.umap(dataset.adata, color="cell_type", save="example_umap")
