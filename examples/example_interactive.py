import sys
import os
from czbenchmarks.datasets.utils import load_dataset

from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

if os.path.exists("/app/model.py"):
    sys.path.insert(0, "/app")
    from model import SCVI as BenchmarkModel
else:
    raise ValueError(
        "Model not found in /app/model.py. This example should be run in a Docker container."
    )

if __name__ == "__main__":
    dataset_list = ["tsv2_heart", "tsv2_large_intestine"]
    datasets = [load_dataset(dataset_name) for dataset_name in dataset_list]

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
