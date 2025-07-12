import logging
import sys
from czbenchmarks.datasets import dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.types import CellRepresentation

# from czbenchmarks.datasets.utils import load_dataset
import numpy as np
from czbenchmarks.tasks import (
    ClusteringTask,
    # EmbeddingTask,
    # MetadataLabelPredictionTask,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")

    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)

    clustering_results = ClusteringTask(random_seed=42).run(
        cell_representation=model_output,
        task_kwargs=dict(
            obs=dataset.adata.obs,
            var=dataset.adata.var,
        ),
        metric_kwargs=dict(input_labels=dataset.labels),
    )

    # task = EmbeddingTask(label_key="cell_type")
    # embedding_results = task.run(dataset)
    # task = EmbeddingTask(label_key="cell_type")
    # embedding_results = task.run(dataset)

    # task = MetadataLabelPredictionTask(label_key="cell_type")
    # prediction_results = task.run(dataset)
    # task = MetadataLabelPredictionTask(label_key="cell_type")
    # prediction_results = task.run(dataset)

    print("Clustering results:")
    print(clustering_results)
    # print("Embedding results:")
    # print(embedding_results)
    # print("Prediction results:")
    # print(prediction_results)
