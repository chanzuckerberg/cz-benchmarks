from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask
from czbenchmarks.datasets.utils import load_dataset, list_available_datasets
import pickle

if __name__ == "__main__":
    all_datasets = list_available_datasets()

    all_results = {}

    for dataset_name in all_datasets:
        print(dataset_name)
        if not dataset_name.startswith("tsv2_"):
            continue
        
        dataset = load_dataset(dataset_name)
        
        for model_name in ["SCVI", "SCGPT"]:
            dataset = run_inference(model_name, dataset)
            
        task = ClusteringTask(label_key="cell_type")
        clustering_results = task.run(dataset)
        
        task = EmbeddingTask(label_key="cell_type")
        embedding_results = task.run(dataset)
        
        task = MetadataLabelPredictionTask(label_key="cell_type")
        metadata_results_cell_type = task.run(dataset)
        
        task = MetadataLabelPredictionTask(label_key="sex")
        metadata_results_sex = task.run(dataset)  
            
        all_results[dataset_name] = {
            "clustering_results": clustering_results,
            "embedding_results": embedding_results,
            "metadata_results_cell_type": metadata_results_cell_type,
            "metadata_results_sex": metadata_results_sex
        }
        
        pickle.dump(all_results, open("all_results.pkl", "wb"))