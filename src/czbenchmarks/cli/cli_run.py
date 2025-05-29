import argparse
import itertools
import json
import logging
import os
import sys
import yaml

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

from czbenchmarks import runner
import czbenchmarks.cli.utils as cli_utils
from czbenchmarks.cli.types import (
    CacheOptions,
    DatasetDetail,
    ModelArgs,
    ModelArgsDict,
    ModelDetail,
    TaskArgs,
    TaskResult,
    TaskType,
)
from czbenchmarks.constants import PROCESSED_DATASETS_CACHE_PATH
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks import exceptions
from czbenchmarks.metrics.types import MetricResult
from czbenchmarks.models import utils as model_utils
from czbenchmarks.models.types import ModelType
from czbenchmarks.tasks import utils as task_utils
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.single_cell.cross_species import CrossSpeciesIntegrationTask
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask
from czbenchmarks import utils


log = logging.getLogger(__name__)


VALID_OUTPUT_FORMATS = ["json", "yaml"]
DEFAULT_OUTPUT_FORMAT = "json"


def _parse_str_to_bool(val):
    """allows us to have arguments that can be explicitly turned on or off
    instead of being limited to store_true and store_false
    """
    if val.lower() in ("true", "1", "yes", "on"):
        return True
    elif val.lower() in ("false", "0", "no", "off"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")


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
        help="One or more dataset names (from datasets.yaml).",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        choices=task_utils.TASK_NAMES,
        help="One or more tasks to run.",
    )
    parser.add_argument(
        "--output-format",
        "-fmt",
        choices=VALID_OUTPUT_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format for results (ignored if --output-file specifies a valid file extension)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to file or directory to save results (default is stdout)",
    )
    parser.add_argument(
        "--remote-cache-url",
        help=(
            "AWS S3 URL prefix for caching embeddings and storing output "
            "(example: s3://cz-benchmarks-example/). Files will be stored "
            "underneath the current --version number. This alone will not "
            "trigger any caching behavior unless one or more of the "
            "--remote-cache-download-embeddings, --remote-cache-upload-embeddings "
            "or --remote-cache-upload-results flags are specified."
        ),
    )
    parser.add_argument(
        "--remote-cache-download-embeddings",
        action="store_true",
        help=(
            "If specified, download embeddings from the remote cache to "
            "PROCESSED_DATASETS_CACHE_PATH if local versions do not exist "
            "or are older than those in the remote cache. Only embeddings "
            "matching the current version will be downloaded."
        ),
        default=False,
    )
    parser.add_argument(
        "--remote-cache-upload-embeddings",
        action="store_true",
        help=(
            "Upload any processed embeddings produced to the remote cache, overwriting "
            "any that may already exist there for the current version. They will be "
            "stored under s3://<remote_cache_url>/<version>/processed-datasets/*.dill"
        ),
        default=False,
    )
    parser.add_argument(
        "--remote-cache-upload-results",
        action="store_true",
        help=(
            "Upload the results to the remote cache. This allows results "
            "to be shared across instances. They will be stored under "
            "s3://<remote_cache_url>/<version>/results/<timestamp>-<random_hex>.json"
        ),
        default=False,
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

    # Extra arguments for AIDO model
    parser.add_argument(
        "--aido-model-variant",
        nargs="*",
        choices=["aido_cell_3m", "aido_cell_10m", "aido_cell_100m"],
        default="aido_cell_3m",
        help="Variant of the aido model to use. Default is aido_cell_3m",
    )

    parser.add_argument(
        "--aido-batch-size",
        type=int,
        nargs="*",
        help="Batch size for AIDO model inference (optional)",
    )

    # universal task arguments
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility (optional)",
    )

    # Extra arguments for clustering task
    parser.add_argument(
        "--clustering-task-label-key",
        help="Label key to use for clustering task",
    )
    parser.add_argument(
        "--clustering-task-set-baseline",
        action="store_true",
        help="Preprocess dataset and set PCA embedding as the BASELINE model output in the dataset",
    )

    # Extra arguments for embedding task
    parser.add_argument(
        "--embedding-task-label-key",
        help="Label key to use for embedding task",
    )
    parser.add_argument(
        "--embedding-task-set-baseline",
        action="store_true",
        help="Preprocess dataset and set PCA embedding as the BASELINE model output in the dataset",
    )

    # Extra arguments for label prediction task
    parser.add_argument(
        "--label-prediction-task-label-key",
        help="Label key to use for label prediction task",
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
        "--label-prediction-task-min-class-size",
        type=int,
        help="Minimum samples required per class (optional)",
    )

    # Extra arguments for integration task
    parser.add_argument(
        "--integration-task-label-key",
        help="Label key to use for integration task",
    )
    parser.add_argument(
        "--integration-task-set-baseline",
        action="store_true",
        help="Use raw gene expression matrix as features for classification (instead of embeddings)",
    )
    parser.add_argument(
        "--integration-task-batch-key",
        help="Key to access batch labels in metadata",
    )

    # Extra arguments for cross species integration task
    parser.add_argument(
        "--cross-species-task-label-key",
        help="Label key to use for cross species integration task",
    )

    # Extra arguments for perturbation task
    parser.add_argument(
        "--perturbation-task-set-baseline",
        action="store_true",
        help="Use mean and median predictions as the BASELINE model output in the dataset",
    )
    parser.add_argument(
        "--perturbation-task-baseline-gene-pert",
        type=str,
        help="Gene perturbation to use for baseline",
    )

    # Advanced feature: define multiple batches of jobs using JSON
    parser.add_argument(
        "--batch-json",
        "-b",
        nargs="+",
        default=[],
        help='Override CLI arguments from the given JSON, e.g. \'{"output_file": "..."}\'. Can be set multiple times to run complex "batch" jobs.',
    )

    parser.add_argument(
        "--batch-random-seeds",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Batch together multiple runs that are identical except for a random seed. "
            "Shortcut for --batch-json '{\"random_seed\": 1}' '{\"random_seed\": 2}' ..."
        ),
    )

    parser.add_argument(
        "--batch-aggregate-metrics",
        nargs="?",
        default=None,  # lets us have a different default depending on batch random seed
        const=True,  # if specified alone without a value, acts like store_true
        type=_parse_str_to_bool,  # lets user explicitly specify true or false to override default
        help=(
            "Aggregate the metrics from tasks that use the same model and dataset if they are run multiple times in a batch job. "
            "For example, if --batch-random-seeds is used to specify multiple random seeds for a task, the metrics reported will "
            "provide an estimate of the mean performance at that task and the uncertainty. "
            "Defaults to true if multiple --batch-random-seeds are specified, but can be disabled with --batch-aggregate-metrics=False"
        ),
    )


def main(parsed_args: argparse.Namespace) -> None:
    """
    Execute a series of tasks using multiple models on a collection of datasets.

    This function handles the benchmarking process by iterating over the specified datasets,
    running inference with the provided models to generate results, and running the tasks to evaluate
    the generated outputs.
    """
    task_results: list[TaskResult] = []
    batch_args = parse_batch_json(
        parsed_args.batch_json, parsed_args.batch_random_seeds
    )
    if parsed_args.batch_aggregate_metrics is None:
        # if they specified --batch-aggregate-metrics explicitly, use that
        # but if they didn't, default to true if multiple seeds were specified, otherwise false
        parsed_args.batch_aggregate_metrics = len(parsed_args.batch_random_seeds) > 1

    cache_options = CacheOptions.from_args(parsed_args)

    for batch_idx, batch_dict in enumerate(batch_args):
        log.info(f"Starting batch {batch_idx + 1}/{len(batch_args)}")

        args = deepcopy(parsed_args)
        for batch_key, batch_val in batch_dict.items():
            setattr(args, batch_key, batch_val)

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
        if "cross_species" in args.tasks:
            task_args.append(
                parse_task_args("cross_species", CrossSpeciesIntegrationTask, args)
            )

        # Run the tasks
        task_result = run(
            dataset_names=args.datasets,
            model_args=model_args,
            task_args=task_args,
            cache_options=cache_options,
        )
        task_results.extend(task_result)

    if args.batch_aggregate_metrics:
        task_results = cli_utils.aggregate_task_results(task_results)

    # Write the results to the specified output
    write_results(
        task_results,
        cache_options=cache_options,
        output_format=args.output_format,
        output_file=args.output_file,
    )


def run(
    dataset_names: list[str],
    model_args: list[ModelArgs],
    task_args: list[TaskArgs],
    cache_options: CacheOptions,
) -> list[TaskResult]:
    """
    Run a set of tasks against a set of datasets. Runs inference if any `model_args` are specified.
    """
    log.info(
        f"Starting benchmarking batch for {len(dataset_names)} datasets, {len(model_args)} models, and {len(task_args)} tasks"
    )
    if not model_args:
        return run_without_inference(dataset_names, task_args)
    return run_with_inference(
        dataset_names, model_args, task_args, cache_options=cache_options
    )


def run_with_inference(
    dataset_names: list[str],
    model_args: list[ModelArgs],
    task_args: list[TaskArgs],
    cache_options: CacheOptions,
) -> list[TaskResult]:
    """
    Execute a series of tasks using multiple models on a collection of datasets.

    This function handles the benchmarking process by iterating over the specified datasets,
    running inference with the provided models to generate results, and running the tasks to evaluate
    the generated outputs.
    """
    task_results: list[TaskResult] = []

    single_dataset_task_names = set(task_utils.TASK_NAMES) - set(
        task_utils.MULTI_DATASET_TASK_NAMES
    )
    single_dataset_tasks: list[TaskArgs] = [
        t for t in task_args if t.name in single_dataset_task_names
    ]
    multi_dataset_tasks: list[TaskArgs] = [
        t for t in task_args if t.name in task_utils.MULTI_DATASET_TASK_NAMES
    ]

    embeddings_for_multi_dataset_tasks: dict[str, BaseDataset] = {}

    # Get all unique combinations of model arguments: each requires a separate inference run
    model_arg_permutations = get_model_arg_permutations(model_args)
    if multi_dataset_tasks and not all(
        len(ma) < 2 for ma in model_arg_permutations.values()
    ):
        raise ValueError(
            "Having multiple model_args for multi-dataset tasks is not supported"
        )

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
                processed_dataset = run_inference_or_load_from_cache(
                    dataset_name,
                    model_name=model_name,
                    model_args=args,
                    cache_options=cache_options,
                )
                # NOTE: accumulating datasets with attached embeddings in memory
                # can be memory intensive
                if multi_dataset_tasks:
                    embeddings_for_multi_dataset_tasks[dataset_name] = processed_dataset

                # Run each single-dataset task against the processed dataset
                for task_arg_idx, task_arg in enumerate(single_dataset_tasks):
                    log.info(
                        f'Starting task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for '
                        f'dataset "{dataset_name}" and model "{model_name}" ({task_arg})'
                    )
                    task_result = run_task(
                        dataset_name, processed_dataset, {model_name: args}, task_arg
                    )
                    task_results.extend(task_result)

    # Run multi-dataset tasks
    embeddings: list[BaseDataset] = list(embeddings_for_multi_dataset_tasks.values())
    for task_arg_idx, task_arg in enumerate(multi_dataset_tasks):
        log.info(
            f'Starting multi-dataset task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for datasets "{dataset_names}"'
        )
        model_args_for_run = {
            model_name: permutation[0]
            for model_name, permutation in model_arg_permutations.items()
            if len(permutation) == 1
        }
        task_result = run_multi_dataset_task(
            dataset_names, embeddings, model_args_for_run, task_arg
        )
        task_results.extend(task_result)

    return task_results


def run_inference_or_load_from_cache(
    dataset_name: str,
    *,
    model_name: str,
    model_args: ModelArgsDict,
    cache_options: CacheOptions,
) -> BaseDataset:
    """
    Load the processed dataset from the cache if it exists, else run inference and save to cache.
    """
    processed_dataset = try_processed_datasets_cache(
        dataset_name,
        model_name=model_name,
        model_args=model_args,
        cache_options=cache_options,
    )
    if processed_dataset:
        log.info("Processed dataset is cached: skipping inference")
        return processed_dataset

    dataset = dataset_utils.load_dataset(dataset_name)
    processed_dataset = runner.run_inference(
        model_name,
        dataset,
        gpu=True,
        **model_args,  # type: ignore [arg-type]
    )

    # if we ran inference, put the embeddings produced into the cache (local and possibly remote)
    set_processed_datasets_cache(
        processed_dataset,
        dataset_name,
        model_name=model_name,
        model_args=model_args,
        cache_options=cache_options,
    )

    return processed_dataset


def run_without_inference(
    dataset_names: list[str], task_args: list[TaskArgs]
) -> list[TaskResult]:
    """
    Run a set of tasks directly against raw datasets without first running model inference.
    """
    task_results: list[TaskResult] = []

    single_dataset_task_names = set(task_utils.TASK_NAMES) - set(
        task_utils.MULTI_DATASET_TASK_NAMES
    )
    single_dataset_tasks: list[TaskArgs] = [
        t for t in task_args if t.name in single_dataset_task_names
    ]
    multi_dataset_tasks: list[TaskArgs] = [
        t for t in task_args if t.name in task_utils.MULTI_DATASET_TASK_NAMES
    ]

    embeddings_for_multi_dataset_tasks: dict[str, BaseDataset] = {}

    for dataset_idx, dataset_name in enumerate(dataset_names):
        log.info(
            f'Processing dataset "{dataset_name}" ({dataset_idx + 1}/{len(dataset_names)})'
        )
        dataset = dataset_utils.load_dataset(dataset_name)
        # NOTE: accumulating datasets with attached embeddings in memory
        # can be memory intensive
        if multi_dataset_tasks:
            embeddings_for_multi_dataset_tasks[dataset_name] = dataset

        for task_arg_idx, task_arg in enumerate(single_dataset_tasks):
            log.info(
                f'Starting task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for dataset "{dataset_name}"'
            )
            task_result = run_task(dataset_name, dataset, {}, task_arg)
            task_results.extend(task_result)

    # Run multi-dataset tasks
    embeddings: list[BaseDataset] = list(embeddings_for_multi_dataset_tasks.values())
    for task_arg_idx, task_arg in enumerate(multi_dataset_tasks):
        log.info(
            f'Starting multi-dataset task "{task_arg.name}" ({task_arg_idx + 1}/{len(task_args)}) for datasets "{dataset_names}"'
        )
        task_result = run_multi_dataset_task(dataset_names, embeddings, {}, task_arg)
        task_results.extend(task_result)

    return task_results


def run_multi_dataset_task(
    dataset_names: list[str],
    embeddings: list[BaseDataset],
    model_args: dict[str, ModelArgsDict],
    task_args: TaskArgs,
) -> list[TaskResult]:
    """
    Run a task and return the results.
    """
    task_results: list[TaskResult] = []

    if task_args.set_baseline:
        raise ValueError("Baseline embedding run not allowed for multi-dataset tasks")

    result: dict[ModelType, list[MetricResult]] = task_args.task.run(embeddings)

    if not isinstance(result, Mapping):
        raise TypeError("Expect a Map ADT for a task result")

    # sorting the dataset_names for the purposes of using it as a
    # cache key and uniform presentation to the user
    dataset_names.sort()

    for model_type, metrics in result.items():
        model_detail = ModelDetail(
            type=model_type, args=model_args.get(model_type.value) or {}
        )
        task_result = TaskResult(
            task_name=task_args.name,
            task_name_display=task_args.task.display_name,
            model=model_detail,
            datasets=[
                DatasetDetail(name=ds_name, organism=ds.organism.value[0])
                for ds_name, ds in zip(dataset_names, embeddings)
            ],
            metrics=metrics,
        )
        task_results.append(task_result)
        log.info(task_result)

    return task_results


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
        task_args.task.set_baseline(dataset, **task_args.baseline_args)

    result: dict[ModelType, list[MetricResult]] = task_args.task.run(dataset)

    if isinstance(result, list):
        raise TypeError("Expected a single task result, got list")

    for model_type, metrics in result.items():
        if model_type == ModelType.BASELINE:
            model_args_to_store = task_args.baseline_args
        else:
            model_args_to_store = model_args.get(model_type.value) or {}

        model_detail = ModelDetail(
            type=model_type,
            args=model_args_to_store,
        )
        task_result = TaskResult(
            task_name=task_args.name,
            task_name_display=task_args.task.display_name,
            model=model_detail,
            datasets=[
                DatasetDetail(name=dataset_name, organism=dataset.organism.value[0])
            ],
            metrics=metrics,
        )
        task_results.append(task_result)
        log.info(task_result)

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
    cache_options: CacheOptions,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    output_file: str | None = None,  # Writes to stdout if None
) -> None:
    """
    Format and write results to the given directory or file.
    """
    results_dict = {
        "czbenchmarks_version": cli_utils.get_version(),
        "args": "czbenchmarks " + " ".join(sys.argv[1:]),
        "task_results": [result.model_dump(mode="json") for result in task_results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    results_str = ""
    if output_format == "json":
        results_str = json.dumps(results_dict, indent=2)
    else:
        results_str = yaml.dump(results_dict)

    if cache_options.remote_cache_url and cache_options.upload_results:
        remote_url = get_result_url_for_remote(cache_options.remote_cache_url)
        try:
            utils.upload_blob_to_remote(
                results_str.encode("utf-8"), remote_url, overwrite_existing=False
            )
        except exceptions.RemoteStorageError:
            log.exception(f"Failed to upload results to {remote_url!r}")
        log.info("Uploaded results to %r", remote_url)

    # Generate a unique filename if we were passed a directory
    if output_file and (os.path.isdir(output_file) or output_file.endswith("/")):
        current_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_file, f"czbenchmarks_results_{current_time}.{output_format}"
        )

    if output_file:
        with open(output_file, "w") as f:
            f.write(results_str)
            f.write("\n")
        log.info("Wrote results to %r", output_file)
    else:
        # Write to stdout if not otherwise specified
        sys.stdout.write(results_str)
        sys.stdout.write("\n")


def get_result_url_for_remote(remote_prefix_url: str) -> str:
    nonce = token_hex(4)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = cli_utils.get_version()
    return f"{remote_prefix_url.rstrip('/')}/{version}/results/{timestamp}-{nonce}.json"


def set_processed_datasets_cache(
    dataset: BaseDataset,
    dataset_name: str,
    *,
    model_name: str,
    model_args: ModelArgsDict,
    cache_options: CacheOptions,
) -> None:
    """
    Write a dataset to the cache
    A "processed" dataset has been run with model inference for the given arguments.
    """
    dataset_filename = get_processed_dataset_cache_filename(
        dataset_name, model_name=model_name, model_args=model_args
    )
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    cache_file = cache_dir / dataset_filename

    try:
        # "Unload" the source data so we only cache the results
        dataset.unload_data()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.serialize(str(cache_file))
        succeeded = True
    except Exception as e:
        # Log the exception, but don't raise if we can't write to the cache for some reason
        log.exception(
            f'Failed to serialize processed dataset to cache "{cache_file}": {e}'
        )
        succeeded = False

    if succeeded and cache_options.upload_embeddings:
        # upload the new embeddings, overwriting any that may already exist
        remote_prefix = get_remote_cache_prefix(cache_options)
        try:
            utils.upload_file_to_remote(
                cache_file, remote_prefix, overwrite_existing=True
            )
            log.info(f"Uploaded processed dataset from {cache_file} to {remote_prefix}")
        except exceptions.RemoteStorageError:
            log.exception("Unable to upload processed dataset to remote cache")

    dataset.load_data()


def try_processed_datasets_cache(
    dataset_name: str,
    *,
    model_name: str,
    model_args: ModelArgsDict,
    cache_options: CacheOptions,
) -> BaseDataset | None:
    """
    Deserialize and return a processed dataset from the cache if it exists, else return None.
    """
    dataset_filename = get_processed_dataset_cache_filename(
        dataset_name, model_name=model_name, model_args=model_args
    )
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    cache_file = cache_dir / dataset_filename

    if cache_options.download_embeddings:
        # check the remote cache and download the file if a local version doesn't
        # exist, or if the remote version is newer than the local version
        remote_url = f"{get_remote_cache_prefix(cache_options)}{dataset_filename}"

        local_modified: datetime | None = None
        remote_modified: datetime | None = None
        if cache_file.exists():
            local_modified = datetime.fromtimestamp(
                cache_file.stat().st_mtime, tz=timezone.utc
            )
        try:
            remote_modified = utils.get_remote_last_modified(
                remote_url, make_unsigned_request=False
            )
        except exceptions.RemoteStorageError:
            # not a great way to handle this, but maybe the cache bucket is not public
            try:
                log.warning(
                    "Unsigned request to remote storage cache failed. Trying signed request."
                )
                remote_modified = utils.get_remote_last_modified(
                    remote_url, make_unsigned_request=True
                )
            except exceptions.RemoteStorageError:
                pass
        if remote_modified is None:
            log.info("Remote cached embeddings don't exist. Skipping download.")
        elif local_modified is not None and (remote_modified <= local_modified):
            log.info(
                f"Remote cached embeddings modified at {remote_modified}. "
                f"Local cache files modified more recently at {local_modified}. "
                "Skipping download."
            )
        else:
            try:
                utils.download_file_from_remote(remote_url, cache_dir)
                log.info(
                    f"Downloaded cached embeddings from {remote_url} to {cache_dir}"
                )
            except exceptions.RemoteStorageError:
                # not a great way to handle this, but maybe the cache bucket is not public
                try:
                    log.warning(
                        "Unsigned request to remote storage cache failed. Trying signed request."
                    )
                    utils.download_file_from_remote(
                        remote_url, cache_dir, make_unsigned_request=False
                    )
                    log.info(
                        f"Downloaded cached embeddings from {remote_url} to {cache_dir}"
                    )
                except exceptions.RemoteStorageError:
                    log.warning(
                        f"Unable to retrieve embeddings from remote cache at {remote_url!r}"
                    )

    if cache_file.exists():
        # Load the original dataset
        dataset = dataset_utils.load_dataset(dataset_name)
        dataset.load_data()

        # Attach the cached results to the dataset
        processed_dataset = BaseDataset.deserialize(str(cache_file))
        dataset._outputs = processed_dataset._outputs
        return dataset

    return None


def get_remote_cache_prefix(cache_options: CacheOptions):
    """get the prefix ending in '/' that the remote processed datasets go under"""
    return f"{cache_options.remote_cache_url.rstrip('/')}/{cli_utils.get_version()}/processed-datasets/"


def get_processed_dataset_cache_filename(
    dataset_name: str, *, model_name: str, model_args: ModelArgsDict
) -> str:
    """
    generate a unique filename for the given dataset and model arguments
    """
    if model_args:
        model_args_str = f"{model_name}_" + "_".join(
            f"{k}-{v}" for k, v in sorted(model_args.items())
        )
    else:
        model_args_str = model_name
    filename = f"{dataset_name}_{model_args_str}.dill"
    return filename


def get_processed_dataset_cache_path(
    dataset_name: str, *, model_name: str, model_args: ModelArgsDict
) -> Path:
    """
    Return a unique file path in the cache directory for the given dataset and model arguments.
    """
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    filename = get_processed_dataset_cache_filename(
        dataset_name, model_name=model_name, model_args=model_args
    )
    return cache_dir / filename


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
    baseline_args: dict[str, Any] = {}

    if args.random_seed is not None:
        task_args["random_seed"] = args.random_seed

    for k, v in vars(args).items():
        if v is not None and k.startswith(prefix):
            trimmed_k = k.removeprefix(prefix)
            if trimmed_k.startswith("baseline_"):
                baseline_args[trimmed_k.removeprefix("baseline_")] = v
            else:
                task_args[trimmed_k] = v

    set_baseline = task_args.pop("set_baseline", False)

    return TaskArgs(
        name=task_name,
        task=TaskCls(**task_args),
        set_baseline=set_baseline,
        baseline_args=baseline_args,
    )


def parse_batch_json(
    batch_json_list: list[str], batch_random_seeds: list[int]
) -> list[dict[str, Any]]:
    """
    Parse the `--batch-json` and `--batch-random-seeds` argument.
    Returns a list of dicts where each entry is a batch of CLI arguments.
    """
    json_batches: list[dict[str, Any]] = []

    if not batch_json_list:
        json_batches = [{}]
    else:
        for batch_json in batch_json_list:
            if not batch_json.strip():
                json_batches.append({})
                continue

            # Load JSON from disk if we were given a valid file path
            if os.path.isfile(batch_json):
                try:
                    with open(batch_json, "r") as f:
                        json_batches.append(json.load(f))
                except Exception as e:
                    raise ValueError(
                        f"Failed to load batch JSON from file {batch_json}: {e}"
                    ) from e
                continue

            # Otherwise treat the input as JSON
            try:
                result = json.loads(batch_json)
                if isinstance(result, list):
                    json_batches.extend(result)
                elif isinstance(result, dict):
                    json_batches.append(result)
                else:
                    raise ValueError(
                        "Invalid batch JSON: input must be a dictionary of CLI arguments"
                    )
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid batch JSON {batch_json}: {e}") from e

    if batch_random_seeds:
        seed_batches = [
            {"random_seed": random_seed} for random_seed in batch_random_seeds
        ]
    else:
        seed_batches = [{}]

    merged_batches: list[dict[str, Any]] = []
    for json_batch in json_batches:
        for seed_batch in seed_batches:
            merged_batches.append({**json_batch, **seed_batch})

    log.info(
        f"Generated {len(merged_batches)} batches from {len(json_batches)} set(s) of arguments and {len(seed_batches)} random seeds"
    )

    return merged_batches
