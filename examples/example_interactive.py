import pathlib
from czbenchmarks.constants import (
    INPUT_DATA_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    get_numbered_path,
)
from czbenchmarks.datasets import BaseDataset
from czbenchmarks.datasets.utils import load_dataset

from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)
from czbenchmarks.utils import get_aws_credentials

from model import SCVI as BenchmarkModel

# # Or dynamically import the model
# model_name = os.environ.get("MODEL_NAME", None)
# if not model_name:
#     raise ValueError("MODEL_NAME environment variable is not set")

# try:
#     module = importlib.import_module("model")
#     BenchmarkModel = getattr(module, model_name.upper())
# except (ImportError, AttributeError) as e:
#     raise ImportError(f"Failed to import {model_name} from model module: {e}")

SERIALIZE_DATASETS = False

if __name__ == "__main__":
    aws_credentials = get_aws_credentials(profile="default")

    # TODO test with multiple datasets
    current_dir = pathlib.Path(__file__).absolute().parent
    dataset = load_dataset("example", config_path=current_dir / "custom_interactive.yaml")
    datasets = dataset if isinstance(dataset, list) else [dataset]

    if SERIALIZE_DATASETS:
        for i, dataset in enumerate(datasets):
            dataset.unload_data()
            input_path = get_numbered_path(INPUT_DATA_PATH_DOCKER, i)
            dataset.serialize(input_path)

    model = BenchmarkModel()

    # FIXME: serializing datasets is coupled to run function. Can this be avoided?
    model_kwargs = {}
    if SERIALIZE_DATASETS:
        model_kwargs = {}
    else:
        model_kwargs = {"datasets": datasets}
    model.run(**model_kwargs)

    if SERIALIZE_DATASETS:
        for i, _ in enumerate(datasets):
            output_path = get_numbered_path(OUTPUT_DATA_PATH_DOCKER, i)
            datasets[i] = BaseDataset.deserialize(output_path)
            datasets[i].load_data()

    task = ClusteringTask(label_key="cell_type")
    clustering_results = task.run(datasets)

    task = EmbeddingTask(label_key="cell_type")
    embedding_results = task.run(datasets)

    task = MetadataLabelPredictionTask(label_key="cell_type")
    prediction_results = task.run(datasets)

    print("Clustering results:")
    print(clustering_results)
    print("Embedding results:")
    print(embedding_results)
    print("Prediction results:")
    print(prediction_results)
