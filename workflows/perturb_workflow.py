import logging
import sys
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import CrossSpeciesIntegrationTask
import os
import json

def save_task_results(task, results, model_name, task_results_dir):
    # usage: save_task_results("cross_species", cross_species_results, "UCE", ".")
    # Save results as JSON
    with open(f"{task_results_dir}/{model_name}_{task}_results.json", "w") as f:
        formatted_results = {
            k.value: [
                {
                    "metric_type": metric.metric_type.value,
                    "value": float(metric.value),
                    "params": metric.params
                }
                for metric in metrics
            ]
            for k, metrics in results.items()
        }
        json.dump(formatted_results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    datasets = ["adamson","norman"]
    model_variants = ["scgenept_ncbi+uniprot_gpt", "scgenept_go_c_gpt_concat", "scgenept_go_all_gpt_concat"]
    model_name = "SCGENEPT"

    for dataset_name in datasets:
        for model_variant in model_variants:
            logging.info(f"Running {model_name} on {dataset_name} with {model_variant} model variant")
            dataset = load_dataset(dataset_name)
            dataset.load_data()
            dataset = run_inference(model_name, dataset, gpu=True, model_variant=model_variant)
            logging.info(f"Inference complete")
            results_dir = f"{model_name}_results"
            os.makedirs(results_dir, exist_ok=True)
            logging.info(f"Saving results to {results_dir}")

            logging.info(f"Running clustering task")
            task = ClusteringTask(label_key="cell_type", random_seed=42)
            clustering_results = task.run(dataset)
            save_task_results("clustering", clustering_results, f"{model_name}_{model_variant}", results_dir)
            logging.info(f"Clustering task complete")

            logging.info(f"Running embedding task")
            task = EmbeddingTask(label_key="cell_type")
            embedding_results = task.run(dataset)
            save_task_results("embedding", embedding_results, f"{model_name}_{model_variant}", results_dir)
            logging.info(f"Embedding task complete")

            logging.info(f"Running metadata label prediction task")
            task = MetadataLabelPredictionTask(label_key="cell_type")
            prediction_results = task.run(dataset)
            save_task_results("prediction", prediction_results, f"{model_name}_{model_variant}", results_dir)
            logging.info(f"Metadata label prediction task complete")

            dataset.serialize(f"{results_dir}/{model_name}_{model_variant}_processed")
            logging.info(f"Dataset serialized {results_dir}/{model_name}_{model_variant}_processed")