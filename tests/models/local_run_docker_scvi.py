from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.runner import ContainerRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask

dataset = SingleCellDataset(
#    "tabula_sapiens_lung.h5ad",
   "Homo_sapiens_ERP132584.h5ad",
    organism=Organism.HUMAN,
)
# Must build local image first
runner = ContainerRunner(
    image="czibench-scvi",
    gpu=True,
)

dataset = runner.run(dataset)

task = ClusteringTask(label_key="cell_type")
dataset, clustering_results = task.run(dataset)

task = EmbeddingTask(label_key="cell_type")
dataset, embedding_results = task.run(dataset)

print(clustering_results)
print(embedding_results)