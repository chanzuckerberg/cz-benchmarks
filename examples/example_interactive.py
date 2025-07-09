import sys
import os
import logging

from czbenchmarks.utils import import_class_from_config
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

logger = logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Import the model class based on _target_ field in the config.yaml file
APP_PATH = os.environ.get("APP_PATH", "/app")
BenchmarkModel = import_class_from_config(os.path.join(APP_PATH, "config.yaml"))

if __name__ == "__main__":
    dataset_list = ["tsv2_bladder", "tsv2_large_intestine"]
    datasets = [
        load_dataset(dataset_name=dataset_name) for dataset_name in dataset_list
    ]

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


# TODO: Modify for new package
