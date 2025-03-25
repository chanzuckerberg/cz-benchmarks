import os
import importlib
from czbenchmarks.datasets.utils import load_dataset

from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)
from czbenchmarks.utils import get_aws_credentials

if __name__ == "__main__":
    # Get model name from environment variable
    model_name = os.environ.get("MODEL_NAME").upper()

    # Dynamically import the model class
    try:
        module = importlib.import_module("model")
        BenchmarkModel = getattr(module, model_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {model_name} from model module: {e}")

    aws_credentials = get_aws_credentials(profile="default")

    # TODO test with multiple datasets
    dataset = load_dataset("example", config_path="custom_interactive.yaml")
    
    model = BenchmarkModel()
    model.run(datasets=dataset)

    task = ClusteringTask(label_key="cell_type")
    clustering_results = task.run(dataset)

    task = EmbeddingTask(label_key="cell_type")
    embedding_results = task.run(dataset)

    task = MetadataLabelPredictionTask(label_key="cell_type")
    prediction_results = task.run(dataset)

    print("Clustering results:")
    print(clustering_results)
    print("Embedding results:")
    print(embedding_results)
    print("Prediction results:")
    print(prediction_results)
