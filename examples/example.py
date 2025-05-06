import logging
import sys
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks.simple import SimpleTask

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    dataset = load_dataset("healthy_human_liver", config_path="src/czbenchmarks/conf/my_datasets.yaml")
    dataset.load_data()
    dataset.validate()
    print(dataset.get_input("ANNDATA"))

    dataset = run_inference("SCVI", dataset)
    print(dataset.get_output("SCVI", "EMBEDDING"))
    
    result = SimpleTask(my_param=10).run(dataset)
    print(result)