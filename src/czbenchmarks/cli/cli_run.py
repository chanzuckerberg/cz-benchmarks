import argparse
import itertools
import json
import logging
import os
import sys
import yaml

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Generic, TypeVar

from czbenchmarks import runner
from czbenchmarks.cli import cli
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks.metrics.types import MetricResult
from czbenchmarks.models import utils as model_utils
from czbenchmarks.models.types import ModelType
from czbenchmarks.tasks import utils as task_utils
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask

log = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = ["json", "yaml"]
DEFAULT_OUTPUT_FORMAT = "json"

TaskType = TypeVar("TaskType", bound=BaseTask)
ModelArgsDict = dict[str, str | int]  # Arguments passed to model inference


class ProcessedDataset(BaseModel):
    model_config = {"arbitrary_types_allowed": True}  # Required to support BaseDataset
    dataset_name: str
    model_args: dict[str, dict[str, str | int]]  # {model_name: {arg_name: arg_value}}
    data: BaseDataset


class ModelArgs(BaseModel):
    name: str  # Upper-case model name e.g. SCVI
    args: dict[str, list[str | int]]  # Args forwarded to the model container


class TaskArgs(BaseModel, Generic[TaskType]):
    model_config = {"arbitrary_types_allowed": True}  # Required to support TaskType
    name: str  # Lower-case task name e.g. embedding
    task: TaskType
    set_baseline: bool


class TaskResult(BaseModel):
    task_name: str
    model_type: str
    dataset_name: str
    model_args: ModelArgsDict
    metrics: list[MetricResult]


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
        nargs="+",
        help="Variant of the geneformer model to use (see docker/geneformer/config.yaml)",
    )

    # Extra arguments for scgenept model
    parser.add_argument(
        "--scgenept-model-variant",
        nargs="+",
        help="Variant of the scgenept model to use (see docker/scgenept/config.yaml)",
    )
    parser.add_argument(
        "--scgenept-gene-pert",
        nargs="+",
        help="Gene perturbation to use for scgenept model",
    )
    parser.add_argument(
        "--scgenept-dataset-name",
        nargs="+",
        help="Dataset name to use for scgenept model",
    )
    parser.add_argument(
        "--scgenept-chunk-size",
        type=int,
        nargs="+",
        help="Chunk size to use for scgenept model",
    )

    # Extra arguments for scgpt model
    parser.add_argument(
        "--scgpt-model-variant",
        nargs="+",
        help="Variant of the scgpt model to use (see docker/scgpt/config.yaml)",
    )

    # Extra arguments for scvi model
    parser.add_argument(
        "--scvi-model-variant",
        nargs="+",
        help="Variant of the scvi model to use (see docker/scvi/config.yaml)",
    )

    # Extra arguments for uce model
    parser.add_argument(
        "--uce-model-variant",
        nargs="+",
        help="Variant of the uce model to use (see docker/uce/config.yaml)",
    )

    # Extra arguments for transcriptformer model
    parser.add_argument(
        "--transcriptformer-model-variant",
        nargs="+",
        choices=["tf-sapiens", "tf-exemplar", "tf-metazoa"],
        help="Variant of the transcriptformer model to use (tf-sapiens, tf-exemplar, tf-metazoa)",
    )
    parser.add_argument(
        "--transcriptformer-batch-size",
        type=int,
        nargs="+",
        help="Batch size for transcriptformer model inference",
    )
    # Extra arguments for clustering task
    parser.add_argument(
        "--clustering-task-label-key",
        nargs="+",
        help="Label key to use for clustering task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--clustering-task-set-baseline",
        action="store_true",
        help="Preprocess dataset and set PCA embedding as the BASELINE model output in the dataset",
    )

    # Extra arguments for embedding task
    parser.add_argument(
        "--embedding-task-label-key",
        nargs="+",
        help="Label key to use for embedding task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--embedding-task-set-baseline",
        action="store_true",
        help="Preprocess dataset and set PCA embedding as the BASELINE model output in the dataset",
    )

    # Extra arguments for label prediction task
    parser.add_argument(
        "--label-prediction-task-label-key",
        nargs="+",
        help="Label key to use for label prediction task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--label-prediction-task-set-baseline",
        action="store_true",
        help="Preprocess dataset and set PCA embedding as the BASELINE model output in the dataset",
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
        nargs="+",
        help="Label key to use for integration task (optional, overrides --label-key)",
    )
    parser.add_argument(
        "--integration-task-set-baseline",
        action="store_true",
        help="Use raw gene expression matrix as features for classification (instead of embeddings)",
    )
    parser.add_argument(
        "--integration-task-batch",
        help="Key to access batch labels in metadata",
    )


def main(args: argparse.Namespace) -> None:
    """
    Execute a series of tasks using multiple models on a collection of datasets.

    This function handles the benchmarking process by iterating over the specified datasets,
    running inference with the provided models to generate results, and running the tasks to evaluate
    the generated outputs.
    """
    # Collect all the arguments that we'll need to pass directly to each model
    model_args: list[ModelArgs] = []
    for model_name in args.models or []:
        model_args.append(parse_model_args(model_name.lower(), args))

    # Collect all the task-related arguments
    task_args: list[TaskArgs] = []
    if "clustering" in args.tasks:
        task_args.append(parse_task_args("clustering", ClusteringTask, args))
    if "embedding" in args.tasks:
        task_args.append(parse_task_args("embedding", EmbeddingTask, args))
    if "label_prediction" in args.tasks:
        task_args.append(
            parse_task_args("label_prediction", MetadataLabelPredictionTask, args)
        )
    if "integration" in args.tasks:
        task_args.append(parse_task_args("integration", BatchIntegrationTask, args))
    if "perturbation" in args.tasks:
        task_args.append(parse_task_args("perturbation", PerturbationTask, args))

    # Run the tasks
    processed_datasets, task_results = run(
        dataset_names=args.datasets,
        model_args=model_args,
        task_args=task_args,
    )

    # Write the results to the specified output
    result_path = write_results(
        task_results,
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
    dataset_names: list[str], model_args: list[ModelArgs], task_args: list[TaskArgs]
) -> tuple[list[ProcessedDataset], list[TaskResult]]:
    """
    Run a set of tasks against a set of datasets. Runs inference if any `model_args` are specified.
    """
    log.info(
        f"Starting benchmarking for {len(dataset_names)} datasets, {len(model_args)} models, and {len(task_args)} tasks"
    )
    if not model_args:
        return run_without_inference(dataset_names, task_args)
    return run_with_inference(dataset_names, model_args, task_args)


def run_with_inference(
    dataset_names: list[str], model_args: list[ModelArgs], task_args: list[TaskArgs]
) -> tuple[list[ProcessedDataset], list[TaskResult]]:
    """
    Execute a series of tasks using multiple models on a collection of datasets.

    This function handles the benchmarking process by iterating over the specified datasets,
    running inference with the provided models to generate results, and running the tasks to evaluate
    the generated outputs.
    """
    task_results: list[TaskResult] = []
    processed_datasets: list[ProcessedDataset] = []

    # Get all unique combinations of model arguments: each requires a separate inference run
    model_arg_permutations = get_model_arg_permutations(model_args)

    for dataset_idx, dataset_name in enumerate(dataset_names):
        log.info(
            f'Processing dataset "{dataset_name}" ({dataset_idx + 1}/{len(dataset_names)})'
        )

        for model_name, model_arg_permutation in model_arg_permutations.items():
            for args_idx, args in enumerate(model_arg_permutation):
                log.info(
                    f'Starting model inference "{model_name}" ({args_idx + 1}/{len(model_arg_permutation)}) '
                    f'for dataset "{dataset_name}"  ({args})'
                )
                processed_data = dataset_utils.load_dataset(dataset_name)
                processed_data = runner.run_inference(
                    model_name,
                    processed_data,
                    gpu=True,
                    **args,  # type: ignore [arg-type]
                )
                processed_dataset = ProcessedDataset(
                    dataset_name=dataset_name,
                    model_args={model_name: args},
                    data=processed_data,
                )
                processed_datasets.append(processed_dataset)

                # Run each task against the processed dataset
                for task_arg_idx, task_arg in enumerate(task_args):
                    log.info(
                        f'Starting task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for '
                        f'dataset "{dataset_name}" and model "{model_name}" ({task_arg})'
                    )
                    task_result = run_task(
                        dataset_name, processed_data, {model_name: args}, task_arg
                    )
                    task_results.extend(task_result)

    return processed_datasets, task_results


def run_without_inference(
    dataset_names: list[str], task_args: list[TaskArgs]
) -> tuple[list[ProcessedDataset], list[TaskResult]]:
    """
    Run a set of tasks directly against raw datasets without first running model inference.
    """
    task_results: list[TaskResult] = []
    processed_datasets: list[ProcessedDataset] = []

    for dataset_idx, dataset_name in enumerate(dataset_names):
        log.info(
            f'Processing dataset "{dataset_name}" ({dataset_idx + 1}/{len(dataset_names)})'
        )
        processed_data = dataset_utils.load_dataset(dataset_name)
        processed_data.load_data()

        processed_dataset = ProcessedDataset(
            dataset_name=dataset_name,
            model_args={},
            data=processed_data,
        )
        processed_datasets.append(processed_dataset)

        for task_arg_idx, task_arg in enumerate(task_args):
            log.info(
                f'Starting task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for dataset "{dataset_name}"'
            )
            task_result = run_task(dataset_name, processed_data, {}, task_arg)
            task_results.extend(task_result)

    return processed_datasets, task_results


def run_task(
    dataset_name: str,
    dataset: BaseDataset,
    model_args: dict[str, ModelArgsDict],
    task_args: TaskArgs,
) -> list[TaskResult]:
    """
    Run a task and return the results.
    """
    task_results: list[TaskResult] = []

    if task_args.set_baseline:
        task_args.task.set_baseline(dataset)

    result: dict[ModelType, list[MetricResult]] = task_args.task.run(dataset)

    if isinstance(result, list):
        raise ValueError("Expected a single task result, got list")

    for model_type, metrics in result.items():
        task_result = TaskResult(
            task_name=task_args.name,
            model_type=model_type.value,
            dataset_name=dataset_name,
            model_args=model_args.get(model_type.value) or {},
            metrics=metrics,
        )
        task_results.append(task_result)

    return task_results


def get_model_arg_permutations(
    model_args: list[ModelArgs],
) -> dict[str, list[ModelArgsDict]]:
    """
    Generate all the "permutations" of model arguments we want to run for each dataset:
    E.g. Running 2 variants of scgenept at 2 chunk sizes results in 4 permutations
    """
    result: dict[str, list[ModelArgsDict]] = defaultdict(list)
    for model_arg in model_args:
        if not model_arg.args:
            result[model_arg.name] = [{}]
            continue
        keys, values = zip(*model_arg.args.items())
        permutations: list[dict[str, str | int]] = [
            {k: v for k, v in zip(keys, permutation)}
            for permutation in itertools.product(*values)
        ]
        result[model_arg.name] = permutations
    return result


def write_results(
    task_results: list[TaskResult],
    *,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    output_file: str | None = None,  # Writes to stdout if None
) -> Path | None:
    """
    Format and write results to the given directory or file.
    Returns the path to the written file, or None if writing only to stdout.
    """
    results_dict = {
        "czbenchmarks_version": cli.get_version(),
        "args": "czbenchmarks " + " ".join(sys.argv[1:]),
        "task_results": [result.model_dump(mode="json") for result in task_results],
    }

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
        result_str = json.dumps(results_dict, indent=2)
    else:
        result_str = yaml.dump(results_dict)

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
    processed_datasets: list[ProcessedDataset], output_dir: str
) -> None:
    """
    Write all processed datasets (with embeddings) to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset in processed_datasets:
        dataset.data.serialize(
            os.path.join(str(output_dir), f"{dataset.dataset_name}.dill")
        )


def parse_model_args(model_name: str, args: argparse.Namespace) -> ModelArgs:
    """
    Populate a ModelArgs instance from the given argparse namespace.
    """
    prefix = f"{model_name.lower()}_"
    model_args: dict[str, Any] = {}
    for k, v in vars(args).items():
        if v is not None and k.startswith(prefix):
            model_args[k.removeprefix(prefix)] = v
    return ModelArgs(name=model_name.upper(), args=model_args)


def parse_task_args(
    task_name: str, TaskCls: type[TaskType], args: argparse.Namespace
) -> TaskArgs:
    """
    Populate a TaskArgs instance from the given argparse namespace.
    """
    prefix = f"{task_name.lower()}_task_"
    task_args: dict[str, Any] = {}

    for k, v in vars(args).items():
        if v is not None and k.startswith(prefix):
            task_args[k.removeprefix(prefix)] = v

    set_baseline = task_args.pop("set_baseline", False)

    return TaskArgs(
        name=task_name, task=TaskCls(**task_args), set_baseline=set_baseline
    )
