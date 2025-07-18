import logging
import sys
import json

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets import dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.utils import list_available_datasets, load_dataset
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
    
    print("Available datasets:", list_available_datasets())
    
    dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")

    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], dataset.adata.shape[0])
    np.save("/tmp/random_model_output.npy", model_output)

    # Initialize all tasks
    clustering_task = ClusteringTask(random_seed=RANDOM_SEED)
    embedding_task = EmbeddingTask(random_seed=RANDOM_SEED)
    prediction_task = MetadataLabelPredictionTask(random_seed=RANDOM_SEED)

    # Compute baseline embeddings for each task
    clustering_baseline = clustering_task.compute_baseline(model_output)
    embedding_baseline = embedding_task.compute_baseline(model_output)
    prediction_baseline = prediction_task.compute_baseline(model_output)

    # Run clustering task with both model output and baseline
    clustering_task_input = ClusteringTaskInput(
        obs=dataset.adata.obs,
        input_labels=dataset.labels,
        use_rep="X",
    )
    clustering_results = clustering_task.run(
        cell_representation=model_output,
        task_input=clustering_task_input,
    )
    clustering_baseline_results = clustering_task.run(
        cell_representation=clustering_baseline,
        task_input=clustering_task_input,
    )

    # Run embedding task with both model output and baseline
    embedding_task_input = EmbeddingTaskInput(
        input_labels=dataset.labels,
    )
    embedding_results = embedding_task.run(
        cell_representation=model_output,
        task_input=embedding_task_input,
    )
    embedding_baseline_results = embedding_task.run(
        cell_representation=embedding_baseline,
        task_input=embedding_task_input,
    )

    # Run prediction task with both model output and baseline
    prediction_task_input = MetadataLabelPredictionTaskInput(
        labels=dataset.labels,
    )
    prediction_results = prediction_task.run(
        cell_representation=model_output,
        task_input=prediction_task_input,
    )
    prediction_baseline_results = prediction_task.run(
        cell_representation=prediction_baseline,
        task_input=prediction_task_input,
    )

    # Combine all results into a single dictionary
    all_results = {
        "clustering": {
            "model": [result.model_dump() for result in clustering_results],
            "baseline": [result.model_dump() for result in clustering_baseline_results],
        },
        "embedding": {
            "model": [result.model_dump() for result in embedding_results],
            "baseline": [result.model_dump() for result in embedding_baseline_results],
        },
        "prediction": {
            "model": [result.model_dump() for result in prediction_results],
            "baseline": [result.model_dump() for result in prediction_baseline_results],
        },
    }

    # Print as nicely formatted JSON
    print(json.dumps(all_results, indent=2, default=str))
