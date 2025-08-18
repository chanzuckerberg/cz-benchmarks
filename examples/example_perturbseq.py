import logging
import sys
# import json

# from czbenchmarks.datasets import dataset
# from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
# Add task inputs
from czbenchmarks.datasets.utils import load_dataset
# from czbenchmarks.tasks.types import CellRepresentation
import argparse

# import numpy as np
import timeit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backed", action="store_true", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    print('Loading dataset')
    results = timeit.timeit(lambda: load_dataset("replogle_k562_essential_perturbpredict", backed=args.backed), number=1)
    print(f"Time to load dataset: {results:.2f} seconds")
    # dataset: SingleCellPerturbationDataset = load_dataset("replogle_k562_essential_perturbpredict")

    # for use_multiprocessing in [False, True]:
    #     for limit_conditions in [500, 1500, None]:
    #         print(f"Loading dataset with limit_conditions={limit_conditions}")
    #         results = timeit.timeit(lambda: dataset.load_data(use_multiprocessing=False, limit_conditions=limit_conditions), number=1)
    #         status_message = f"{limit_conditions} {dataset.adata.shape} {len(dataset.target_genes_to_save)} {results}"
    #         print(status_message)
    #         with open("results.txt", "a") as f:
    #             f.write(f"{status_message}\n")

    # Testing method for storing of outputs -- can remove after finished
    # print('Storing task inputs')
    # task_inputs_dir = dataset.store_task_inputs()
    # print('Writing control matched adata')
    # dataset.control_matched_adata.write_h5ad(task_inputs_dir / "control_matched_adata.h5ad")

    # Test the validation method -- can remove after finished
    # dataset._validate()

    # # NOTE this is a numpy array containing only a matrix
    # model_output: CellRepresentation = np.random.rand(
    #     dataset.adata.shape[0], dataset.adata.shape[1]
    # )
    # np.save("/tmp/random_model_output.npy", model_output)

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
