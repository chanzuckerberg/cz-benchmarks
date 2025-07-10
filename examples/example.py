import logging
import sys
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    dataset = load_dataset("tsv2_lung")
    print(f"Loaded dataset: {dataset.adata}")

    # task = ClusteringTask(label_key="cell_type")
    # clustering_results = task.run(dataset)

    # task = EmbeddingTask(label_key="cell_type")
    # embedding_results = task.run(dataset)

    # task = MetadataLabelPredictionTask(label_key="cell_type")
    # prediction_results = task.run(dataset)

    # print("Clustering results:")
    # print(clustering_results)
    # print("Embedding results:")
    # print(embedding_results)
    # print("Prediction results:")
    # print(prediction_results)


# TODO: Modify for new package
