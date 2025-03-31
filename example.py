import logging
import sys
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    dataset = load_dataset("tsv2_bladder")

    for model_name in ["SCVI", "SCGPT"]:
        dataset = run_inference(model_name, dataset)

        task = ClusteringTask(label_key="cell_type")
        clustering_results = task.run(dataset)

        task = EmbeddingTask(label_key="cell_type")
        embedding_results = task.run(dataset)

        task = MetadataLabelPredictionTask(label_key="cell_type")
        prediction_results = task.run(dataset)

        print("Clustering results:")
        print(clustering_results)
        print("Embedding results:")
        print(embedding_results)
        print("Prediction results:")
        print(prediction_results)
