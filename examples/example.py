import logging
import sys
import json
from czbenchmarks.datasets import dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.types import CellRepresentation

# from czbenchmarks.datasets.utils import load_dataset
import numpy as np
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)
from czbenchmarks.tasks.clustering import ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTaskInput

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")

    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)

    task_input = ClusteringTaskInput(
        obs=dataset.adata.obs,
        input_labels=dataset.labels,
        use_rep="X",
    )
    
    clustering_results = ClusteringTask(random_seed=42).run(
        cell_representation=model_output,
        task_input=task_input,
    )

    embedding_task_input = EmbeddingTaskInput(
        input_labels=dataset.labels,
    )
    embedding_task = EmbeddingTask()
    embedding_results = embedding_task.run(
        cell_representation=model_output,
        task_input=embedding_task_input,
    )

    prediction_task_input = MetadataLabelPredictionTaskInput(
        labels=dataset.labels,
    )
    prediction_task = MetadataLabelPredictionTask()
    prediction_results = prediction_task.run(
        cell_representation=model_output,
        task_input=prediction_task_input,
    )

    # Combine all results into a single dictionary
    all_results = {
        "clustering": [result.model_dump() for result in clustering_results],
        "embedding": [result.model_dump() for result in embedding_results],
        "prediction": [result.model_dump() for result in prediction_results],
    }
    
    # Print as nicely formatted JSON
    print(json.dumps(all_results, indent=2, default=str))
