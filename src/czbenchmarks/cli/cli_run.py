import argparse
import itertools
import logging
import json
import os
import sys
import yaml

from pathlib import Path
from typing import Any, Generic, TypeVar
from dataclasses import asdict, dataclass
from datetime import datetime

from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.models import utils as model_utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks import runner
from czbenchmarks.cli import cli
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks import utils as task_utils
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask

log = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = ["json", "yaml"]
DEFAULT_OUTPUT_FORMAT = "json"

TaskResultType = dict[str, list[dict]]  # {model_name: metrics[]}
DatasetResultType = dict[str, TaskResultType]  # {task_name: TaskResultType}
RunResultType = dict[str, DatasetResultType]  # {dataset_name: DatasetResultType}

TaskType = TypeVar("TaskType", bound=BaseTask)


@dataclass
class RunResult:
    results: RunResultType
    czbenchmarks_version: str
    args: str


@dataclass
class ModelArgs:
    name: str
    args: dict[str, list[str | int]]  # Args forwarded to the model container


@dataclass
class TaskArgs(Generic[TaskType]):
    name: str
    task: TaskType
    set_baseline: bool


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
    model_args: list[ModelArgs] = []
    if args.models:
        if "GENEFORMER" in args.models:
            model_args.append(parse_model_args("geneformer", args))
        if "SCGENEPT" in args.models:
            model_args.append(parse_model_args("scgenept", args))
        if "SCGPT" in args.models:
            model_args.append(parse_model_args("scgpt", args))
        if "SCVI" in args.models:
            model_args.append(parse_model_args("scvi", args))
        if "UCE" in args.models:
            model_args.append(parse_model_args("uce", args))

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
        for model_arg in model_args:
            keys, values = zip(*model_arg.args.items())
            permutations = [
                {k: v for k, v in zip(keys, permutation)}
                for permutation in itertools.product(*values)
            ]

            # Run inference against every permutation of model arguments
            for permutation in permutations:
                log.info(
                    f"Running {model_arg.name} inference on dataset {dataset_name} with args {permutation}"
                )
                processed_dataset = runner.run_inference(
                    model_arg.name, processed_dataset, gpu=True, **permutation
                )

            processed_datasets[dataset_name] = processed_dataset

        # Explicitly oad the dataset into memory if we didn't call `run_inference` above
        if not model_args:
            log.info(f"Loading dataset {dataset_name} into memory")
            processed_dataset.load_data()

        # Run each task on the processed dataset
        for task_arg in task_args:
            if task_arg.set_baseline:
                log.info(
                    f"Setting baseline for {task_arg.name} on dataset {dataset_name}"
                )
                task_arg.task.set_baseline(processed_dataset)
            log.info(f"Running {task_arg.name} on dataset {dataset_name}")
            dataset_results[task_arg.name] = run_task(processed_dataset, task_arg.task)

        # Store the results for this dataset
        run_results[dataset_name] = dataset_results

    return processed_datasets, run_results


def run_task(processed_dataset: BaseDataset, task: TaskType) -> TaskResultType:
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
    results_dict = asdict(results)

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
    processed_datasets: dict[str, BaseDataset], output_dir: str
) -> None:
    """
    Write all processed datasets (with embeddings) to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset in processed_datasets.items():
        dataset.serialize(os.path.join(str(output_dir), f"{dataset_name}.dill"))


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
