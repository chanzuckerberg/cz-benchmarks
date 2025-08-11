import logging
import sys
import json

from czbenchmarks.datasets import dataset
from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
# Add task inputs
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.types import CellRepresentation

import numpy as np


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    dataset: SingleCellPerturbationDataset = load_dataset("replogle_k562_essential_perturbpredict")
    
    # Testing storing of outputs -- can remove after finished
    task_inputs_dir = dataset.store_task_inputs()

    # Testing validation -- can remove after finished
    dataset._validate()

    # NOTE this is a numpy array containing only a matrix
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[0]
    )
    np.save("/tmp/random_model_output.npy", model_output)

    # Initialize task
    # Add task initialization here

    # Compute baseline embeddings for each task
    # Add baseline computation here

    # # Run task with both model output and baseline
    # perturbation_expression_prediction_task_input = ClusteringTaskInput(
    #     obs=dataset.adata.obs,
    #     input_labels=dataset.labels,
    #     use_rep="X",
    # )
    # clustering_results = clustering_task.run(
    #     cell_representation=model_output,
    #     task_input=perturbation_expression_prediction_task_input,
    # )
    # clustering_baseline_results = clustering_task.run(
    #     cell_representation=perturbation_expression_prediction_baseline,
    #     task_input=perturbation_expression_prediction_task_input,
    # )


    # # Combine all results into a single dictionary
    # all_results = {
    #     "perturbation_expression prediction": {
    #         "model": [result.model_dump() for result in clustering_results],
    #         "baseline": [result.model_dump() for result in clustering_baseline_results],
    #     },
    # }

    # # Print as nicely formatted JSON
    # print(json.dumps(all_results, indent=2, default=str))
