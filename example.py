import scanpy as sc

from czibench.datasets.utils import load_dataset
from czibench.runner import ContainerRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask

dataset = load_dataset("example", config_path="custom.yaml")
runner = ContainerRunner(
    image="czibench-scvi:latest",
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
