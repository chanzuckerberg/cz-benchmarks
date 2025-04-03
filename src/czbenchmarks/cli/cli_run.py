import argparse

import logging
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple
from typing_extensions import TypedDict, NotRequired
import yaml
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


class GeneformerModelArgs(TypedDict):
    model_variant: str


class ScgeneptModelArgs(TypedDict):
    model_variant: str
    gene_pert: str
    dataset_name: str
    chunk_size: int


class ScgptModelArgs(TypedDict):
    model_variant: str


class UceModelArgs(TypedDict):
    model_variant: str


class ModelArgs(NamedTuple):
    name: str
    args: GeneformerModelArgs | ScgeneptModelArgs | ScgptModelArgs | UceModelArgs


class ClusteringTaskArgs(TypedDict):
    label_key: str


class EmbeddingTaskArgs(TypedDict):
    label_key: str


class LabelPredictionTaskArgs(TypedDict):
    label_key: str
    n_folds: NotRequired[int]
    seed: NotRequired[int]
    min_class_size: NotRequired[int]


class IntegrationTaskArgs(TypedDict):
    label_key: str
    batch_key: NotRequired[str]


class TaskArgs(NamedTuple):
    name: str
    args: (
        ClusteringTaskArgs
        | EmbeddingTaskArgs
        | LabelPredictionTaskArgs
        | IntegrationTaskArgs
    )
    cls: (
        type[ClusteringTask]
        | type[EmbeddingTask]
        | type[MetadataLabelPredictionTask]
        | type[BatchIntegrationTask]
    )


class TaskResult(TypedDict):
    results: dict[str, list[dict]]
    args: (
        ClusteringTaskArgs
        | EmbeddingTaskArgs
        | LabelPredictionTaskArgs
        | IntegrationTaskArgs
    )


class RunResult(TypedDict):
    results: dict[str, dict[str, TaskResult]]  # {dataset_name: {task_name: TaskResult}}
    czbenchmarks_version: str
    args: str


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
    if "geneformer" in args.models:
        model_args.append(
            ModelArgs(
                name="geneformer",
                args=GeneformerModelArgs(
                    model_variant=args.geneformer_model_variant,
                ),
            )
        )
    if "scgenept" in args.models:
        model_args.append(
            ModelArgs(
                name="scgenept",
                args=ScgeneptModelArgs(
                    model_variant=args.scgenept_model_variant,
                    gene_pert=args.scgenept_gene_pert,
                    dataset_name=args.scgenept_dataset_name,
                    chunk_size=args.scgenept_chunk_size,
                ),
            )
        )
    if "scgpt" in args.models:
        model_args.append(
            ModelArgs(
                name="scgpt",
                args=ScgptModelArgs(
                    model_variant=args.scgpt_model_variant,
                ),
            )
        )
    if "uce" in args.models:
        model_args.append(
            ModelArgs(
                name="uce",
                args=UceModelArgs(
                    model_variant=args.uce_model_variant,
                ),
            )
        )

    task_args: list[TaskArgs] = []
    if "clustering" in args.tasks:
        task_args.append(
            TaskArgs(
                name="clustering",
                args=ClusteringTaskArgs(
                    label_key=args.clustering_task_label_key or args.label_key
                ),
                cls=ClusteringTask,
            )
        )
    if "embedding" in args.tasks:
        task_args.append(
            TaskArgs(
                name="embedding",
                args=EmbeddingTaskArgs(
                    label_key=args.embedding_task_label_key or args.label_key
                ),
                cls=EmbeddingTask,
            )
        )
    if "label_prediction" in args.tasks:
        task_args.append(
            TaskArgs(
                name="label_prediction",
                args=LabelPredictionTaskArgs(
                    label_key=args.label_prediction_task_label_key or args.label_key,
                    n_folds=args.label_prediction_task_n_folds,
                    seed=args.label_prediction_task_seed,
                    min_class_size=args.label_prediction_task_min_class_size,
                ),
                cls=MetadataLabelPredictionTask,
            )
        )
    if "integration" in args.tasks:
        task_args.append(
            TaskArgs(
                name="integration",
                args=IntegrationTaskArgs(
                    label_key=args.integration_task_label_key or args.label_key,
                    batch_key=args.integration_task_batch,
                ),
                cls=BatchIntegrationTask,
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
        if isinstance(args.save_embeddings, str):
            dirname = args.save_embeddings
        elif result_path:
            dirname = str(result_path.parent / f"{result_path.stem}_processed_datasets")
        write_processed_datasets(processed_datasets, dirname)


def run(
    *,
    dataset_names: list[str],
    model_args: list[ModelArgs],
    task_args: list[TaskArgs],
) -> tuple[dict[str, BaseDataset], dict[str, dict[str, TaskResult]]]:
    """
    Run a set of tasks for a set of models on a set of datasets.
    Returns a tuple of (BaseDataset[], {dataset_name: {task_name: TaskResult}}).
    """
    dataset_results: dict[str, dict[str, TaskResult]] = {}
    processed_datasets: dict[str, BaseDataset] = {}

    for dataset_name in dataset_names:
        task_results: dict[str, TaskResult] = {}

        log.info(f"Loading dataset {dataset_name}")
        processed_dataset = dataset_utils.load_dataset(dataset_name)

        # Run inference against this dataset for each model to generate embeddings
        for model_name, model_arg in model_args:
            log.info(f"Running {model_name} inference on dataset {dataset_name}")
            processed_dataset = runner.run_inference(
                model_name, processed_dataset, gpu=True, **model_arg
            )
            processed_datasets[dataset_name] = processed_dataset

        # Run each task on the processed dataset
        for task_name, task_arg, TaskCls in task_args:
            task_results[task_name] = run_task(TaskCls, processed_dataset, task_arg)

        # Store the results for this dataset
        dataset_results[dataset_name] = task_results

    return processed_datasets, dataset_results


def run_task(
    TaskCls: type[ClusteringTask]
    | type[EmbeddingTask]
    | type[MetadataLabelPredictionTask]
    | type[BatchIntegrationTask],
    processed_dataset: BaseDataset,
    task_args: ClusteringTaskArgs
    | EmbeddingTaskArgs
    | LabelPredictionTaskArgs
    | IntegrationTaskArgs,
) -> TaskResult:
    """
    Run a task and return the results.
    """
    log.info("Running %s with args: %s", TaskCls.__name__, task_args)
    task = TaskCls(**task_args)
    results = task.run(processed_dataset)

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


def write_processed_datasets(
    processed_datasets: dict[str, BaseDataset], output_dir: str
) -> None:
    """
    Write all processed datasets (with embeddings) to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset in processed_datasets.items():
        dataset.serialize(os.path.join(str(output_dir), f"{dataset_name}.dill"))
