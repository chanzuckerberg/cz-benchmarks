import logging
import sys
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import (
    PerturbationTask,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    dataset = load_dataset("adamson_perturb")

    dataset = run_inference("SCGENEPT", dataset, gene_pert="TMED2+ctrl")

    task = PerturbationTask()
    perturbation_results = task.run(dataset)

    print(perturbation_results)


# TODO: Modify for new package
