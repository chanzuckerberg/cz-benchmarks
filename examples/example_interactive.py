import sys
import pathlib

from czbenchmarks.datasets.utils import load_dataset

from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

sys.path.insert(0, "/app") # Set to location of model.py
from model import SCVI as BenchmarkModel

if __name__ == "__main__":
    # TODO test with multiple datasets
    current_dir = pathlib.Path(__file__).absolute().parent
    dataset = load_dataset("example", config_path=current_dir / "custom_interactive.yaml")
    datasets = dataset if isinstance(dataset, list) else [dataset]

    model = BenchmarkModel()
    model.run(datasets=datasets)

    task = ClusteringTask(label_key="cell_type")
    clustering_results = task.run(datasets)

    task = EmbeddingTask(label_key="cell_type")
    embedding_results = task.run(datasets)

    task = MetadataLabelPredictionTask(label_key="cell_type")
    prediction_results = task.run(datasets)

    print("Clustering results:")
    print(clustering_results)
    print("Embedding results:")
    print(embedding_results)
    print("Prediction results:")
    print(prediction_results)
