from czibench.runners.mlflow_runner import MLflowModelRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask
from czibench.datasets.utils import load_dataset
import scanpy as sc

dataset = load_dataset("example", config_path="custom.yaml")

runner = MLflowModelRunner(model_resource_url="model-serving/runtimes/mlflow/scvi/runtime")
dataset = runner.run(dataset)
dataset.load_data()
print(dataset.output_embedding)

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


