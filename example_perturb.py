from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import (
    PerturbationTask,
)

if __name__ == "__main__":
    dataset = load_dataset("adamson_perturb")

    for model_name in ["SCGENEPT"]:
        dataset = run_inference(model_name, dataset, gene_pert="TMED2+ctrl")

    task = PerturbationTask()
    perturbation_results = task.run(dataset)

    print(perturbation_results)