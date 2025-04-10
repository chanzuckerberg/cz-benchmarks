import sys
import os
import logging

from czbenchmarks.utils import import_class_from_config
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks import PerturbationTask

logger = logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Import the model class based on _target_ field in the config.yaml file
APP_PATH = os.environ.get("APP_PATH", "/app")
BenchmarkModel = import_class_from_config(os.path.join(APP_PATH, "config.yaml"))

if __name__ == "__main__":
    # Set the gene perturbation flags
    sys.argv += ["--gene_pert", "TMED2+ctrl"]

    dataset_list = ["adamson_perturb"]
    datasets = [
        load_dataset(dataset_name=dataset_name) for dataset_name in dataset_list
    ]

    model = BenchmarkModel()
    model.run(datasets=datasets)

    task = PerturbationTask()
    perturbation_results = task.run(datasets)

    print(perturbation_results)
