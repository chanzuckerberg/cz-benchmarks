import argparse
import itertools
import json
import logging
import os
import sys
import yaml

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
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


@dataclass
class ProcessedDataset:
    dataset_name: str
    model_args: dict[str, dict[str, str | int]]  # {model_name: {arg_name: arg_value}}
    data: BaseDataset


@dataclass
class ModelArgs:
    name: str  # Upper-case model name e.g. SCVI
    args: dict[str, list[str | int]]  # Args forwarded to the model container


@dataclass
class TaskArgs(Generic[TaskType]):
    name: str  # Lower-case task name e.g. embedding
    task: TaskType
    set_baseline: bool


@dataclass
class TaskResult:
    task_name: str
    model_type: str
    dataset_name: str
    model_args: ModelArgsDict
    metrics: list[MetricResult]

    def to_dict(self) -> dict[str, Any]:
        task_result_dict = asdict(self)
        task_result_dict["metrics"] = [
            metric.model_dump(mode="json") for metric in self.metrics
        ]
        return task_result_dict


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

    # Extra arguments for clustering task
    parser.add_argument(
        "--clustering-task-label-key",
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
    Run a set of tasks for a set of models on a set of datasets.
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
    *,
    dataset_names: list[str],  # One per dataset
    model_args: list[ModelArgs],  # One per model
    task_args: list[TaskArgs],  # One per task
) -> tuple[list[ProcessedDataset], list[TaskResult]]:
    """
    Run a set of tasks for a set of models on a set of datasets.
    """
    task_results: list[TaskResult] = []
    processed_datasets: list[ProcessedDataset] = []

    for dataset_name in dataset_names:
        model_permutations = get_model_arg_permutations(model_args)

        # Run inference against a single dataset if there is only one set of arguments per model:
        # A single dataset can support multiple models, but not multiple variants of the same model
        if all(len(permutations) == 1 for permutations in model_permutations.values()):
            args = {k: v[0] for k, v in model_permutations.items()}
            processed_dataset, task_results = run_tasks_for_shared_dataset(
                dataset_name, args, task_args
            )
            processed_datasets.append(processed_dataset)
            task_results.extend(task_results)

        # To run inference with multiple variants of the same model we must load separate datasets:
        # A single dataset can support multiple models, but not multiple variants of the same model
        else:
            processed_datasets, task_results = run_tasks_for_dataset_variants(
                dataset_name, model_permutations, task_args
            )
            processed_datasets.extend(processed_datasets)
            task_results.extend(task_results)

    return processed_datasets, task_results


def run_tasks_for_shared_dataset(
    dataset_name: str,
    model_args: dict[str, ModelArgsDict],  # {model_name: ModelArgsDict}
    task_args: list[TaskArgs],
) -> tuple[ProcessedDataset, list[TaskResult]]:
    """
    Run tasks against a shared dataset: used when there is a single set of arguments per model.
    A single dataset can support multiple models, but not multiple variants of the same model.
    """
    task_results: list[TaskResult] = []

    # Dataset is loaded once and shared across all models and tasks
    log.info(f"Loading dataset {dataset_name}")
    processed_data = dataset_utils.load_dataset(dataset_name)

    # Run inference for each model and attach embeddings to the dataset
    for model_name, model_arg in model_args.items():
        log.info(
            f"Running {model_name} inference on dataset {dataset_name} with args {model_arg}"
        )
        processed_data = runner.run_inference(
            model_name,
            processed_data,
            gpu=True,
            **model_arg,  # type: ignore [arg-type]
        )

    # Run each task on the shared dataset
    for task_arg in task_args:
        log.info(f"Running {task_arg.name} on dataset {dataset_name}")
        task_result = run_task(dataset_name, processed_data, model_args, task_arg)
        task_results.extend(task_result)

    # Wrap the processed data with some extra mdetadata
    processed_dataset = ProcessedDataset(
        dataset_name=dataset_name,
        model_args=model_args,
        data=processed_data,
    )
    return processed_dataset, task_results


def run_tasks_for_dataset_variants(
    dataset_name: str,
    model_arg_permutations: dict[str, list[dict[str, str | int]]],
    task_args: list[TaskArgs],
) -> tuple[list[ProcessedDataset], list[TaskResult]]:
    """
    Run tasks against separate dataset variants: used when there are multiple sets of arguments (permutations) per model.
    A single dataset can support multiple models, but not multiple variants of the same model.
    """
    task_results: list[TaskResult] = []
    processed_datasets: list[ProcessedDataset] = []

    for model_name, model_args in model_arg_permutations.items():
        for args in model_args:
            # Create a new dataset for each set of model arguments
            log.info(f"Loading dataset {dataset_name}")
            processed_data = dataset_utils.load_dataset(dataset_name)

            log.info(
                f"Running {model_name} inference on dataset {dataset_name} with args {args}"
            )
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

            for task_arg in task_args:
                log.info(
                    f"Running {task_arg.name} on dataset {dataset_name} variant {args}"
                )
                task_result = run_task(
                    dataset_name, processed_data, {model_name: args}, task_arg
                )
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
        dataset.load_data()
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
        "task_results": [result.to_dict() for result in task_results],
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
