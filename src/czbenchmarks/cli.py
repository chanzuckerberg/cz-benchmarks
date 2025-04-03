"""
czbenchmarks CLI

Usage:
    czbenchmarks list <list_type>
    czbenchmarks run --models <model_name> --datasets <dataset_name> --TASK_NAME-task-label-key <label_key>
        [--clustering-task-label-key <label_key>]
        [--embedding-task-label-key <label_key>]
        [--prediction-task-label-key <label_key>]
        [--integration-task-label-key <label_key>]
        [--save-embeddings <output_dir>]
"""

import logging
import json
import os
import sys
import tomli
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from typing_extensions import TypedDict, NotRequired
import yaml
from datetime import datetime
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.models import utils as model_utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks import runner
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.integration import BatchIntegrationTask

log = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = ["json", "yaml"]
DEFAULT_OUTPUT_FORMAT = "json"


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


class TaskResult(TypedDict):
    results: dict[str, list[dict]]
    args: (
        ClusteringTaskArgs
        | EmbeddingTaskArgs
        | PredictionTaskArgs
        | IntegrationTaskArgs
    )


class DatasetResult(TypedDict):
    dataset_name: str
    clustering_task: NotRequired[TaskResult]
    embedding_task: NotRequired[TaskResult]
    prediction_task: NotRequired[TaskResult]
    integration_task: NotRequired[TaskResult]


class RunResult(TypedDict):
    results: list[DatasetResult]
    czbenchmarks_version: str
    args: str


def run(
    *,
    model_names: list[str],
    dataset_names: list[str],
    clustering_task_args: ClusteringTaskArgs | None = None,
    embedding_task_args: EmbeddingTaskArgs | None = None,
    prediction_task_args: PredictionTaskArgs | None = None,
    integration_task_args: IntegrationTaskArgs | None = None,
) -> tuple[dict[str, BaseDataset], list[DatasetResult]]:
    """
    Run a set of tasks for a set of models on a set of datasets.
    Returns a tuple of (processed_datasets, results).
    """
    results: list[DatasetResult] = []
    processed_datasets: dict[str, BaseDataset] = {}

    for dataset_name in dataset_names:
        result = DatasetResult(dataset_name=dataset_name)

        log.info(f"Loading dataset {dataset_name}")
        embeddings = dataset_utils.load_dataset(dataset_name)

        # Run inference against this dataset for each model to generate embeddings
        for model_name in model_names:
            log.info(f"Running {model_name} inference on dataset {dataset_name}")
            embeddings = runner.run_inference(model_name, embeddings, gpu=True)
            processed_datasets[dataset_name] = embeddings

        if clustering_task_args:
            result["clustering_task"] = run_task(
                ClusteringTask, embeddings, clustering_task_args
            )
        if embedding_task_args:
            result["embedding_task"] = run_task(
                EmbeddingTask, embeddings, embedding_task_args
            )
        if prediction_task_args:
            result["prediction_task"] = run_task(
                MetadataLabelPredictionTask, embeddings, prediction_task_args
            )
        if integration_task_args:
            result["integration_task"] = run_task(
                BatchIntegrationTask, embeddings, integration_task_args
            )
        results.append(result)

    return processed_datasets, results


def run_task(
    TaskCls: type[ClusteringTask]
    | type[EmbeddingTask]
    | type[MetadataLabelPredictionTask]
    | type[BatchIntegrationTask],
    embeddings: BaseDataset,
    task_args: ClusteringTaskArgs
    | EmbeddingTaskArgs
    | PredictionTaskArgs
    | IntegrationTaskArgs,
) -> TaskResult:
    """
    Run a task and return the results.
    """
    log.info("Running %s with args: %s", TaskCls.__name__, task_args)
    task = TaskCls(**task_args)
    results = task.run(embeddings)

    if isinstance(results, list):
        raise ValueError("Expected a single task result, got list")

    # Serialize the result to a json-compatible dict
    json_results = {
        k.value: [v.model_dump(mode="json") for v in val] for k, val in results.items()
    }
    return TaskResult(results=json_results, args=task_args)


def write_results(
    results: RunResult,
    *,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    output_file: str | None = None,  # Writes to stdout if None
) -> Path | None:
    """
    Format and write results to the given directory or file.
    Returns the path to the written file, or None if writing only to stdout.
    """
    # Get the intended format/extension
    if output_file and output_file.endswith(".json"):
        output_format = "json"
    elif output_file and (
        output_file.endswith(".yaml") or output_file.endswith(".yml")
    ):
        output_format = "yaml"
    elif output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Invalid output format: {output_format}")

    # Dump the results to a string
    result_str = ""
    if output_format == "json":
        result_str = json.dumps(results, indent=2)
    else:
        result_str = yaml.dump(results)

    # Write to stdout if not otherwise specified
    if not output_file:
        sys.stdout.write(f"{result_str}\n")
        return None

    # Generate a unique filename if we were passed a directory
    if os.path.isdir(output_file) or output_file.endswith("/"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_file, f"czbenchmarks_results_{current_time}.{output_format}"
        )

    # Write the results to the specified file
    with open(output_file, "w") as f:
        f.write(f"{result_str}\n")

    log.info("Wrote results to %s", output_file)
    return Path(output_file)


def write_embeddings(
    embeddings: dict[str, BaseDataset], output_dir: str | Path
) -> None:
    """
    Write all processed datasets (with embeddings) to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset in embeddings.items():
        dataset.serialize(os.path.join(str(output_dir), f"{dataset_name}.dill"))


def get_version() -> str:
    """
    Get the current version of the czbenchmarks library.
    """
    try:
        return version("czbenchmarks")
    except PackageNotFoundError:
        log.debug(
            "Package `czbenchmarks` is not installed: fetching version info from pyproject.toml"
        )

    # In development this lib might not be installed as a package so try loading from pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


def main() -> None:
    """Entry point for the czbenchmarks CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="czbenchmark: A command-line utility for single-cell benchmark tasks."
    )
    parser.add_argument(
        "--version",
        help="Show version number and exit",
        action="version",
        version=f"%(prog)s {get_version()}",
    )
    parser.add_argument(
        "--log-level",
        "-ll",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level (default is info)",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)

    # Run parser
    run_parser = subparsers.add_parser("run", help="Run a set of tasks.")
    run_parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=model_utils.list_available_models(),
        help="One or more model names (from models.yaml).",
        required=True,
    )
    run_parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        choices=dataset_utils.list_available_datasets(),
        required=True,
        help="One or more dataset names (from datasets.yaml).",
    )
    run_parser.add_argument(
        "--output-format",
        "-fmt",
        choices=VALID_OUTPUT_FORMATS,
        default="yaml",
        help="Output format for results (ignored if --output-file specifies a valid file extension)",
    )
    run_parser.add_argument(
        "--output-file",
        "-o",
        help="Path to file or directory to save results (default is stdout)",
    )
    run_parser.add_argument(
        "--save-embeddings",
        "-s",
        nargs="?",
        const=True,
        help="Save processed datasets to the specified directory, or save to an auto-generated directory if this flag is set with no arguments (datasets are not saved by default)",
    )

    # Run parser clustering task
    run_parser.add_argument(
        "--clustering-task-label-key",
        help="Key to access ground truth labels in metadata",
    )

    # Run parser embedding task
    run_parser.add_argument(
        "--embedding-task-label-key",
        help="Key to access ground truth labels in metadata",
    )

    # Run parser prediction task
    run_parser.add_argument(
        "--prediction-task-label-key",
        help="Key to access ground truth labels in metadata",
    )
    run_parser.add_argument(
        "--prediction-task-n-folds",
        type=int,
        help="Number of cross-validation folds (optional)",
    )
    run_parser.add_argument(
        "--prediction-task-seed",
        type=int,
        help="Random seed for reproducibility (optional)",
    )
    run_parser.add_argument(
        "--prediction-task-min-class-size",
        type=int,
        help="Minimum samples required per class (optional)",
    )

    # Run parser integration task
    run_parser.add_argument(
        "--integration-task-label-key",
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
        choices=["datasets", "models"],
        help="List available datasets or models.",
    )

    # Parse arguments to dict
    try:
        args = vars(parser.parse_args())
        logging.basicConfig(level=args["log_level"].upper(), stream=sys.stdout)
    except argparse.ArgumentError as e:
        parser.error(str(e))
    except SystemExit:
        raise

    if args["action"] == "list":
        if args["list_type"] == "datasets":
            sys.stdout.write(" ".join(dataset_utils.list_available_datasets()))
        elif args["list_type"] == "models":
            sys.stdout.write(" ".join(model_utils.list_available_models()))
        sys.stdout.write("\n")
        sys.exit(0)

    if args["action"] == "run":
        clustering_task_args: ClusteringTaskArgs | None = None
        embedding_task_args: EmbeddingTaskArgs | None = None
        prediction_task_args: PredictionTaskArgs | None = None
        integration_task_args: IntegrationTaskArgs | None = None

        if args.get("clustering_task_label_key"):
            pre = "clustering_task_"
            clustering_task_args = ClusteringTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )
        if args.get("embedding_task_label_key"):
            pre = "embedding_task_"
            embedding_task_args = EmbeddingTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )
        if args.get("prediction_task_label_key"):
            pre = "prediction_task_"
            prediction_task_args = PredictionTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )
        if args.get("integration_task_label_key"):
            pre = "integration_task_"
            integration_task_args = IntegrationTaskArgs(
                **{k.removeprefix(pre): v for k, v in args.items() if k.startswith(pre)}
            )

        # Run the tasks
        embeddings, results = run(
            model_names=args["models"],
            dataset_names=args["datasets"],
            clustering_task_args=clustering_task_args,
            embedding_task_args=embedding_task_args,
            prediction_task_args=prediction_task_args,
            integration_task_args=integration_task_args,
        )

        # Append some extra metadata before writing the results
        run_result = RunResult(
            results=results,
            czbenchmarks_version=get_version(),
            args=" ".join(sys.argv[1:]),
        )

        # Write the results to the specified output
        result_path = write_results(
            run_result,
            output_format=args["output_format"],
            output_file=args.get("output_file"),
        )

        # Optionally write processed datasets to disk
        if args.get("save_embeddings"):
            dirname = datetime.now().strftime("czbenchmarks_embeddings_%Y%m%d_%H%M%S")
            if isinstance(args.get("save_embeddings"), str):
                dirname = args["save_embeddings"]
            elif result_path:
                dirname = result_path.parent / f"{result_path.stem}_embeddings"
            write_embeddings(embeddings, dirname)


if __name__ == "__main__":
    main()
