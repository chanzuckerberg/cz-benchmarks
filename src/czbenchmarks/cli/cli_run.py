import argparse
import logging
import json
import os
import sys
import yaml

from pathlib import Path
from typing import Any, NamedTuple
from typing_extensions import TypedDict
from datetime import datetime

from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.models import utils as model_utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks import runner
from czbenchmarks.cli import cli
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks import utils as task_utils

log = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = ["json", "yaml"]
DEFAULT_OUTPUT_FORMAT = "json"

TaskResultType = dict[str, list[dict]]  # {model_name: metrics[]}
DatasetResultType = dict[str, TaskResultType]  # {task_name: TaskResultType}
RunResultType = dict[str, DatasetResultType]  # {dataset_name: DatasetResultType}


class RunResult(TypedDict):
    results: RunResultType
    czbenchmarks_version: str
    args: str


class ModelArgs(NamedTuple):
    name: str
    args: dict[str, Any]  # Args forwarded to the model container


class TaskArgs(NamedTuple):
    name: str
    task: (
        ClusteringTask
        | EmbeddingTask
        | MetadataLabelPredictionTask
        | BatchIntegrationTask
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add run command arguments to the parser.
    """

    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=model_utils.list_available_models(),
        help="One or more model names (from models.yaml).",
        required=True,
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        choices=dataset_utils.list_available_datasets(),
        required=True,
        help="One or more dataset names (from datasets.yaml).",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        choices=task_utils.TASK_NAMES,
        required=True,
        help="One or more tasks to run.",
    )
    parser.add_argument(
        "--label-key",
        "-l",
        required=True,
        help="The dataset column to use as the label key, e.g. `cell_type` (can be overridden per-task).",
    )
    parser.add_argument(
        "--output-format",
        "-fmt",
        choices=VALID_OUTPUT_FORMATS,
        default="yaml",
        help="Output format for results (ignored if --output-file specifies a valid file extension)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to file or directory to save results (default is stdout)",
    )
    parser.add_argument(
        "--save-processed-datasets",
        "-s",
        nargs="?",
        const=True,
        help="Save processed datasets to the specified directory, or save to an auto-generated directory if this flag is set with no arguments (datasets are not saved by default)",
    )

    # Extra arguments for geneformer model
    parser.add_argument(
        "--geneformer-model-variant",
        help="Variant of the geneformer model to use (see docker/geneformer/config.yaml)",
    )

    # Extra arguments for scgenept model
    parser.add_argument(
        "--scgenept-model-variant",
        help="Variant of the scgenept model to use (see docker/scgenept/config.yaml)",
    )
    parser.add_argument(
        "--scgenept-gene-pert",
        help="Gene perturbation to use for scgenept model",
    )
    parser.add_argument(
        "--scgenept-dataset-name",
        help="Dataset name to use for scgenept model",
    )
    parser.add_argument(
        "--scgenept-chunk-size",
        type=int,
        help="Chunk size to use for scgenept model",
    )

    # Extra arguments for scgpt model
    parser.add_argument(
        "--scgpt-model-variant",
        help="Variant of the scgpt model to use (see docker/scgpt/config.yaml)",
    )

    # Extra arguments for scvi model
    parser.add_argument(
        "--scvi-model-variant",
        help="Variant of the scvi model to use (see docker/scvi/config.yaml)",
    )

    # Extra arguments for uce model
    parser.add_argument(
        "--uce-model-variant",
        help="Variant of the uce model to use (see docker/uce/config.yaml)",
    )

    # Extra arguments for clustering task
    parser.add_argument(
        "--clustering-task-label-key",
        help="Label key to use for clustering task (optional, overrides --label-key)",
    )

    # Extra arguments for embedding task
    parser.add_argument(
        "--embedding-task-label-key",
        help="Label key to use for embedding task (optional, overrides --label-key)",
    )

    # Extra arguments for prediction task
    parser.add_argument(
        "--label-prediction-task-label-key",
        help="Label key to use for label prediction task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--label-prediction-task-n-folds",
        type=int,
        help="Number of cross-validation folds (optional)",
    )
    parser.add_argument(
        "--label-prediction-task-seed",
        type=int,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--label-prediction-task-min-class-size",
        type=int,
        help="Minimum samples required per class (optional)",
    )

    # Extra arguments for integration task
    parser.add_argument(
        "--integration-task-label-key",
        help="Label key to use for integration task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--integration-task-batch",
        help="Key to access batch labels in metadata",
    )


def main(args: argparse.Namespace) -> None:
    """
    Run a set of tasks for a set of models on a set of datasets.
    """
    model_args: list[ModelArgs] = []
    if "GENEFORMER" in args.models:
        model_args.append(
            ModelArgs(name="GENEFORMER", args=parse_model_args("geneformer_", args)),
        )
    if "SCGENEPT" in args.models:
        model_args.append(
            ModelArgs(name="SCGENEPT", args=parse_model_args("scgenept_", args)),
        )
    if "SCGPT" in args.models:
        model_args.append(
            ModelArgs(name="SCGPT", args=parse_model_args("scgpt_", args)),
        )
    if "SCVI" in args.models:
        model_args.append(
            ModelArgs(name="SCVI", args=parse_model_args("scvi_", args)),
        )
    if "UCE" in args.models:
        model_args.append(
            ModelArgs(name="UCE", args=parse_model_args("uce_", args)),
        )

    task_args: list[TaskArgs] = []
    if "clustering" in args.tasks:
        task_args.append(
            TaskArgs(
                name="clustering",
                task=ClusteringTask(**parse_task_args("clustering_task_", args)),
            )
        )
    if "embedding" in args.tasks:
        task_args.append(
            TaskArgs(
                name="embedding",
                task=EmbeddingTask(**parse_task_args("embedding_task_", args)),
            )
        )
    if "label_prediction" in args.tasks:
        task_args.append(
            TaskArgs(
                name="label_prediction",
                task=MetadataLabelPredictionTask(
                    **parse_task_args("label_prediction_task_", args)
                ),
            )
        )
    if "integration" in args.tasks:
        task_args.append(
            TaskArgs(
                name="integration",
                task=BatchIntegrationTask(**parse_task_args("integration_task_", args)),
            )
        )

    # Run the tasks
    processed_datasets, dataset_results = run(
        dataset_names=args.datasets,
        model_args=model_args,
        task_args=task_args,
    )

    # Append some extra metadata before writing the results
    run_result = RunResult(
        results=dataset_results,
        czbenchmarks_version=cli.get_version(),
        args=" ".join(sys.argv[1:]),
    )

    # Write the results to the specified output
    result_path = write_results(
        run_result,
        output_format=args.output_format,
        output_file=args.output_file,
    )

    # Optionally write processed datasets to disk
    if args.save_processed_datasets:
        dirname = datetime.now().strftime(
            "czbenchmarks_processed_datasets_%Y%m%d_%H%M%S"
        )
        if args.save_processed_datasets and isinstance(
            args.save_processed_datasets, str
        ):
            dirname = args.save_processed_datasets
        elif result_path:
            dirname = str(result_path.parent / f"{result_path.stem}_processed_datasets")
        write_processed_datasets(processed_datasets, dirname)


def run(
    *,
    dataset_names: list[str],
    model_args: list[ModelArgs],
    task_args: list[TaskArgs],
) -> tuple[dict[str, BaseDataset], RunResultType]:
    """
    Run a set of tasks for a set of models on a set of datasets.
    """
    run_results: RunResultType = {}
    processed_datasets: dict[str, BaseDataset] = {}

    for dataset_name in dataset_names:
        dataset_results: dict[str, TaskResultType] = {}

        log.info(f"Loading dataset {dataset_name}")
        processed_dataset = dataset_utils.load_dataset(dataset_name)

        # Run inference against this dataset for each model to generate embeddings
        for model_name, args in model_args:
            log.info(f"Running {model_name} inference on dataset {dataset_name}")
            processed_dataset = runner.run_inference(
                model_name, processed_dataset, gpu=True, **args
            )
            processed_datasets[dataset_name] = processed_dataset

        # Run each task on the processed dataset
        for task_name, task in task_args:
            log.info(f"Running {task_name} on dataset {dataset_name}")
            dataset_results[task_name] = run_task(processed_dataset, task)

        # Store the results for this dataset
        run_results[dataset_name] = dataset_results

    return processed_datasets, run_results


def run_task(
    processed_dataset: BaseDataset,
    task: ClusteringTask
    | EmbeddingTask
    | MetadataLabelPredictionTask
    | BatchIntegrationTask,
) -> TaskResultType:
    """
    Run a task and return the results.
    """
    results = task.run(processed_dataset)

    if isinstance(results, list):
        raise ValueError("Expected a single task result, got list")

    # Serialize the result to a json-compatible dict
    json_results = {
        k.value: [v.model_dump(mode="json") for v in val] for k, val in results.items()
    }
    return json_results


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


def write_processed_datasets(
    processed_datasets: dict[str, BaseDataset], output_dir: str
) -> None:
    """
    Write all processed datasets (with embeddings) to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset in processed_datasets.items():
        dataset.serialize(os.path.join(str(output_dir), f"{dataset_name}.dill"))


def parse_model_args(prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    """
    For a dict of `args`, strip the given prefix and return all matching non-null entries.
    """
    result: dict[str, Any] = {}
    for k, v in vars(args).items():
        if v is not None and k.startswith(prefix):
            result[k.removeprefix(prefix)] = v
    return result


def parse_task_args(prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    """
    For a dict of `args`, strip the given prefix and return all matching non-null entries.
    Also sets "label_key" which is required for all tasks.
    """
    result: dict[str, Any] = {"label_key": args.label_key}  # "label_key" is required
    for k, v in vars(args).items():
        if v is not None and k.startswith(prefix):
            result[k.removeprefix(prefix)] = v
    return result
