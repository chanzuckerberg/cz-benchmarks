from __future__ import annotations

import json
import logging
import os
from typing import Optional, get_args

import click
import numpy as np
import pandas as pd

from czbenchmarks.constants import RANDOM_SEED as DEFAULT_SEED
from czbenchmarks.datasets import load_dataset
from czbenchmarks.datasets.utils import load_custom_dataset
from czbenchmarks.tasks.task import TASK_REGISTRY, TaskParameter

from .runner import run_task
from .specs import BenchmarkRunSpec

logger = logging.getLogger(__name__)


def load_numpy_array_from_path(file_path: str) -> np.ndarray:
    """Load a numpy array from various file formats.

    Supports .npy, .npz, .csv, and .tsv files. For .npz files, uses the first
    array key if multiple arrays are present.

    Args:
        file_path: Path to the file (can start with file:// protocol)

    Returns:
        Loaded numpy array

    Raises:
        ValueError: If file path is invalid, file doesn't exist, or format is unsupported
    """
    logger.debug(f"Loading numpy array from path: {file_path}")
    if file_path.startswith("@"):
        raise ValueError("File path cannot be an AnnData reference")

    actual_path = file_path
    if actual_path.startswith("file://"):
        actual_path = actual_path[7:]
    if not os.path.exists(actual_path):
        raise ValueError(f"File does not exist: {actual_path}")

    file_extension = os.path.splitext(actual_path)[1].lower()
    logger.debug(f"Loading array with extension: {file_extension}")
    if file_extension == ".npy":
        return np.load(actual_path)
    if file_extension == ".npz":
        npz_data = np.load(actual_path)
        keys = list(npz_data.keys())
        if not keys:
            raise ValueError(f"No arrays found in .npz file: {actual_path}")
        if len(keys) > 1:
            logger.info(f"Multiple arrays in .npz file. Using first key: '{keys[0]}'")
        return npz_data[keys[0]]
    if file_extension in (".csv", ".tsv"):
        separator = "\t" if file_extension == ".tsv" else ","
        dataframe = pd.read_csv(actual_path, sep=separator, header=None)
        return dataframe.values
    raise ValueError(f"Unsupported file format: {file_extension}")


def _build_option_from_param(
    param: TaskParameter, is_baseline: bool = False
) -> click.Option:
    """Create a click.Option from a TaskParameter with proper type mapping.

    Args:
        param: TaskParameter to convert to Click option
        is_baseline: Whether this is a baseline parameter

    Returns:
        click.Option configured for the parameter
    """
    prefix = "--baseline-" if is_baseline else "--"
    long_name = f"{prefix}{param.name.replace('_', '-')}"

    click_type = click.STRING
    if param.type is bool:
        click_type = click.BOOL
    elif param.type is int:
        click_type = click.INT
    elif param.type is float:
        click_type = click.FLOAT

    param_str = str(param.type)
    if "Literal" in param_str:
        try:
            choices = get_args(param.type)
            if choices:
                click_type = click.Choice([str(c) for c in choices])
        except Exception:
            pass

    help_text = param.help_text
    if not param.required and param.default is not None:
        help_text += f" (Default: {param.default})"
    if param.is_multiple:
        help_text += " [multiple]"

    default_value = param.default if not param.is_multiple else None
    if is_baseline:
        default_value = None

    kwargs = {
        "type": click_type,
        "multiple": param.is_multiple,
        "default": default_value,
        "help": help_text,
        "required": param.required and param.default is None,
    }

    if click_type is click.BOOL:
        kwargs["is_flag"] = True
        kwargs["default"] = bool(param.default) if param.default is not None else False
        return click.Option([long_name], **kwargs)

    option_names = [long_name]
    if param.name == "organism_list" and not is_baseline:
        option_names.append("--organisms")

    return click.Option(option_names, **kwargs)


def add_shared_cli_options():
    """Create shared CLI options for all task commands.

    Returns:
        List of click.Option decorators for shared parameters like dataset,
        cell representation path, baseline computation, random seed, and output file.
    """

    def mutually_exclusive_callback(ctx, param, value):
        if not value or (isinstance(value, tuple) and len(value) == 0):
            return value
        other_option = "dataset" if param.name == "user_dataset" else "user_dataset"
        other_value = ctx.params.get(other_option)

        if other_value and (
            (isinstance(other_value, tuple) and len(other_value) > 0) or other_value
        ):
            raise click.UsageError(
                f"'{param.opts[0]}' and '--{other_option}' are mutually exclusive."
            )
        return value

    return [
        click.option(
            "-d",
            "--dataset",
            multiple=True,
            help="Dataset name available in czbenchmarks.",
            callback=mutually_exclusive_callback,
        ),
        click.option(
            "-u",
            "--user-dataset",
            multiple=True,
            help='Path to a user .h5ad file as JSON: \'{"dataset_class": "SingleCellDataset", "organism": "Organism.Human", "path": "~/mydata.h5ad"}\'.',
            callback=mutually_exclusive_callback,
        ),
        click.option(
            "-c",
            "--cell-representation-path",
            multiple=True,
            help="Path to embedding array (.npy/.npz/.csv/.tsv) or AnnData reference like @X, @obsm:X_pca.",
            default=("@X",),
            show_default=True,
        ),
        click.option(
            "-b",
            "--compute-baseline",
            is_flag=True,
            default=False,
            help="If set, compute and evaluate the task baseline.",
        ),
        click.option(
            "-r",
            "--random-seed",
            type=int,
            default=DEFAULT_SEED,
            show_default=True,
            help="Set a random seed for reproducibility.",
        ),
        click.option(
            "-o",
            "--output-file",
            type=click.Path(dir_okay=False, writable=True, resolve_path=True),
            help="Write JSON results to a file.",
        ),
    ]


@click.group(
    name="run",
    context_settings=dict(help_option_names=["-h", "--help"]),
    help="""Run benchmark tasks on dataset and model output embeddings""",
)
def run():
    pass


def add_dynamic_task_command(task_name: str):
    """Create a dynamic Click command for a specific task using TaskRegistry.

    Leverages the enhanced TaskRegistry to:
    - Build task-specific CLI options automatically
    - Validate task and baseline inputs before execution
    - Handle multi-dataset tasks with proper parameter alignment
    - Provide rich help text from task metadata

    Args:
        task_name: The task identifier from TASK_REGISTRY

    Returns:
        click.Command configured for the task
    """
    task_info = TASK_REGISTRY.get_task_info(task_name)

    def task_execution_handler(**cli_kwargs):
        """Handle execution of a task with validated parameters."""
        logger.debug(
            f"CLI execution handler called for task '{task_name}' with {len(cli_kwargs)} parameters"
        )

        random_seed: int = cli_kwargs.pop("random_seed", DEFAULT_SEED)
        output_file_path: Optional[str] = cli_kwargs.pop("output_file", None)

        cli_kwargs["task_key"] = task_name

        try:
            spec = BenchmarkRunSpec.from_cli_args(cli_kwargs)
        except Exception as e:
            raise click.ClickException(f"Error parsing arguments: {e}")

        is_multi_dataset = task_info.requires_multiple_datasets

        try:
            adata_list = []

            for dataset_key in spec.czb_dataset_keys:
                logger.debug(f"Loading dataset: {dataset_key}")
                dataset = load_dataset(dataset_key)
                if not hasattr(dataset, "adata"):
                    raise click.ClickException(
                        f"Dataset '{dataset_key}' does not provide an `.adata` attribute."
                    )
                adata_list.append(dataset.adata)
                logger.debug(
                    f"Dataset '{dataset_key}' loaded successfully. AnnData shape: {dataset.adata.shape}"
                )

            for user_dataset_spec in spec.user_datasets:
                logger.debug(f"Loading user dataset from: {user_dataset_spec.path}")
                dataset = load_local_dataset(
                    dataset_class=user_dataset_spec.dataset_class,
                    organism=user_dataset_spec.organism,
                    path=str(user_dataset_spec.path),
                )
                if not hasattr(dataset, "adata"):
                    raise click.ClickException(
                        "User dataset does not provide an `.adata` attribute."
                    )
                adata_list.append(dataset.adata)
                logger.debug(
                    f"User dataset loaded successfully. AnnData shape: {dataset.adata.shape}"
                )

            if not adata_list:
                raise click.ClickException(
                    "No datasets loaded. Specify at least one dataset using --dataset or --user-dataset."
                )

            if not is_multi_dataset:
                adata_input = adata_list[0]
            else:
                adata_input = adata_list

            logger.debug(f"Loaded {len(adata_list)} dataset(s)")

        except Exception as dataset_error:
            raise click.ClickException(f"Error loading dataset: {dataset_error}")

        try:
            cell_rep_list = []

            cell_rep_paths = spec.cell_representations
            if (
                is_multi_dataset
                and len(cell_rep_paths) == 1
                and "," in cell_rep_paths[0]
            ):
                cell_rep_paths = [path.strip() for path in cell_rep_paths[0].split(",")]

            for idx, crp in enumerate(cell_rep_paths):
                if crp.startswith("@"):
                    cell_rep_list.append(crp)
                    logger.debug(
                        f"Cell representation {idx} is an AnnData reference: {crp}"
                    )
                else:
                    try:
                        arr = load_numpy_array_from_path(crp)
                        cell_rep_list.append(arr)
                        logger.debug(
                            f"Cell representation {idx} loaded as array with shape: {arr.shape}"
                        )
                    except Exception as load_error:
                        raise click.ClickException(
                            f"Failed to load cell representation from '{crp}': {load_error}"
                        )

            if not is_multi_dataset:
                cell_representation_input = cell_rep_list[0] if cell_rep_list else "@X"
            else:
                if not cell_rep_list:
                    raise click.ClickException(
                        f"Multi-dataset task '{task_name}' requires at least one cell representation."
                    )
                cell_representation_input = cell_rep_list

            logger.debug(f"Loaded {len(cell_rep_list)} cell representation(s)")

        except click.ClickException:
            raise
        except Exception as rep_error:
            raise click.ClickException(
                f"Error loading cell representation: {rep_error}"
            )

        try:
            logger.info(f"Executing task '{task_name}'...")
            logger.debug(f"Task parameters: {spec.task_inputs}")
            logger.debug(f"Baseline parameters: {spec.baseline_args}")
            logger.debug(f"Run baseline: {spec.run_baseline}")

            execution_results = run_task(
                task_name=task_name,
                adata_input=adata_input,
                cell_representation_input=cell_representation_input,
                run_baseline=spec.run_baseline,
                baseline_params=spec.baseline_args or {},
                task_params=spec.task_inputs or {},
                random_seed=random_seed,
            )
        except Exception as execution_error:
            raise click.ClickException(f"Task execution failed: {execution_error}")

        json_output = json.dumps(execution_results, indent=2, default=str)
        if output_file_path:
            with open(output_file_path, "w") as output_file:
                output_file.write(json_output)
            logger.info(f"Results written to {output_file_path}")
        else:
            click.echo(json_output)

    task_command = click.command(
        name=task_name,
        help=task_info.description,
        context_settings={"help_option_names": ["-h", "--help"]},
    )(task_execution_handler)

    for cli_option in add_shared_cli_options():
        task_command = cli_option(task_command)

    for param_info in reversed(list(task_info.task_params.values())):
        option = _build_option_from_param(param_info, is_baseline=False)
        task_command.params.append(option)

    for param_info in reversed(list(task_info.baseline_params.values())):
        option = _build_option_from_param(param_info, is_baseline=True)
        task_command.params.append(option)

    return task_command


for task_name in TASK_REGISTRY.list_tasks():
    run.add_command(add_dynamic_task_command(task_name))


__all__ = ["run"]
