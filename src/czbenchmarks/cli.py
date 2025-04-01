import logging
import sys
from typing_extensions import TypedDict, NotRequired
from czbenchmarks.datasets import utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks.runner import ContainerRunner
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.integration import BatchIntegrationTask

log = logging.getLogger(__name__)


class ClusteringTaskArgs(TypedDict):
    label_key: str


class EmbeddingTaskArgs(TypedDict):
    label_key: str


class PredictionTaskArgs(TypedDict):
    label_key: str
    n_folds: NotRequired[int]
    seed: NotRequired[int]
    min_class_size: NotRequired[int]


class IntegrationTaskArgs(TypedDict):
    label_key: str
    batch_key: NotRequired[str]


def run(
    *,
    model_names: list[str],
    dataset_names: list[str],
    clustering_task_args: ClusteringTaskArgs | None = None,
    embedding_task_args: EmbeddingTaskArgs | None = None,
    prediction_task_args: PredictionTaskArgs | None = None,
    integration_task_args: IntegrationTaskArgs | None = None,
) -> None:
    """
    Run a set of tasks for a set of models on a set of datasets.
    """

    # Load datasets
    datasets: dict[str, BaseDataset] = {}
    for dataset_name in dataset_names:
        datasets[dataset_name] = utils.load_dataset(dataset_name)

    # Load each model
    for model_name in model_names:
        runner = ContainerRunner(model_name)

        for dataset_name, dataset in datasets.items():
            embeddings = runner.run(dataset)

            if clustering_task_args:
                clustering_task = ClusteringTask(**clustering_task_args)
                clustering_results = clustering_task.run(embeddings)
                log.info(
                    f"Clustering results for {model_name} on {dataset_name}:\n\n{clustering_results}\n"
                )

            if embedding_task_args:
                embedding_task = EmbeddingTask(**embedding_task_args)
                embedding_results = embedding_task.run(embeddings)
                log.info(
                    f"Embedding results for {model_name} on {dataset_name}:\n\n{embedding_results}\n"
                )

            if prediction_task_args:
                prediction_task = MetadataLabelPredictionTask(**prediction_task_args)
                prediction_results = prediction_task.run(embeddings)
                log.info(
                    f"Prediction results for {model_name} on {dataset_name}:\n\n{prediction_results}\n"
                )

            if integration_task_args:
                integration_task = BatchIntegrationTask(**integration_task_args)
                integration_results = integration_task.run(embeddings)
                log.info(
                    f"Integration results for {model_name} on {dataset_name}:\n\n{integration_results}\n"
                )

    log.info("success")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    import argparse

    parser = argparse.ArgumentParser(
        description="czbenchmark: A command-line utility for single-cell benchmark tasks."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Run parser
    run_parser = subparsers.add_parser("run", help="Run a set of tasks.")
    run_parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="One or more model names (from models.yaml).",
        required=True,
    )
    run_parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        required=True,
        help="One or more dataset names (from datasets.yaml).",
    )

    # Run parser clustering task
    run_parser.add_argument(
        "--clustering-task-label",
        help="Key to access ground truth labels in metadata",
    )

    # Run parser embedding task
    run_parser.add_argument(
        "--embedding-task-label",
        help="Key to access ground truth labels in metadata",
    )

    # Run parser prediction task
    run_parser.add_argument(
        "--prediction-task-label",
        help="Key to access ground truth labels in metadata",
    )
    run_parser.add_argument(
        "--prediction-task-n-folds",
        type=int,
        help="Number of cross-validation folds",
    )
    run_parser.add_argument(
        "--prediction-task-seed",
        type=int,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--prediction-task-min-class-size",
        type=int,
        help="Minimum samples required per class",
    )

    # Run parser integration task
    run_parser.add_argument(
        "--integration-task-label",
        help="Key to access ground truth labels in metadata",
    )
    run_parser.add_argument(
        "--integration-task-batch",
        help="Key to access batch labels in metadata",
    )

    # List parser
    list_parser = subparsers.add_parser("list", help="List datasets/models/tasks.")
    list_parser.add_argument(
        "list_type",
        type=str,
        choices=["datasets", "models", "tasks"],
        help="What you want to list: 'datasets', 'models', or 'tasks'.",
    )

    # Parse arguments to dict
    try:
        args = vars(parser.parse_args())
    except argparse.ArgumentError as e:
        parser.error(str(e))
    except SystemExit:
        raise

    if args["action"] == "list":
        raise NotImplementedError("This commmand is not yet implemented")

    if args["action"] == "run":
        clustering_task_args: ClusteringTaskArgs | None = None
        embedding_task_args: EmbeddingTaskArgs | None = None
        prediction_task_args: PredictionTaskArgs | None = None
        integration_task_args: IntegrationTaskArgs | None = None

        if args.get("clustering_task_label"):
            pre = "clustering_task_"
            clustering_task_args = ClusteringTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )

        if args.get("embedding_task_label"):
            pre = "embedding_task_"
            embedding_task_args = EmbeddingTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )

        if args.get("prediction_task_label"):
            pre = "prediction_task_"
            prediction_task_args = PredictionTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )

        if args.get("integration_task_label"):
            pre = "integration_task_"
            integration_task_args = IntegrationTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )

        run(
            model_names=args["models"],
            dataset_names=args["datasets"],
            clustering_task_args=clustering_task_args,
            embedding_task_args=embedding_task_args,
            prediction_task_args=prediction_task_args,
            integration_task_args=integration_task_args,
        )
