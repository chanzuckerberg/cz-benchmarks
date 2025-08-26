from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import click
import numpy as np
import pandas as pd

from czbenchmarks.constants import RANDOM_SEED as DEFAULT_SEED
from czbenchmarks.tasks.task import TASK_REGISTRY
from .runner import run_task
from czbenchmarks.datasets import load_dataset


def load_numpy_array_from_path(file_path: str) -> np.ndarray:
    if not isinstance(file_path, str):
        raise ValueError("File path must be a string")
    if file_path.startswith("@"):
        raise ValueError("File path cannot be an AnnData reference")

    actual_path = file_path
    if actual_path.startswith("file://"):
        actual_path = actual_path[7:]
    if not os.path.exists(actual_path):
        raise ValueError(f"File does not exist: {actual_path}")

    file_extension = os.path.splitext(actual_path)[1].lower()
    if file_extension == ".npy":
        return np.load(actual_path)
    if file_extension == ".npz":
        npz_data = np.load(actual_path)
        first_array_key = list(npz_data.keys())[0]
        return npz_data[first_array_key]
    if file_extension in (".csv", ".tsv"):
        separator = "\t" if file_extension == ".tsv" else ","
        dataframe = pd.read_csv(actual_path, sep=separator, header=None)
        return dataframe.values
    raise ValueError(f"Unsupported file format: {file_extension}")


def convert_cli_parameter(param_value: str, param_info) -> Any:
    if param_value is None:
        return None

    param_type = param_info.type
    param_str = str(param_type).lower()

    if param_type is int or "int" == param_str:
        return int(param_value)
    elif param_type is float or "float" == param_str:
        return float(param_value)
    elif param_type is bool or "bool" == param_str:
        return param_value.lower() in ("true", "1", "yes", "on")
    else:
        return param_value


def add_shared_cli_options():
    return [
        click.option(
            "--dataset",
            required=True,
            help="Dataset identifier.",
        ),
        click.option(
            "--cell-representation-path",
            help="Path to embedding arrays (.npy/.npz/.csv/.tsv) or AnnData reference like @X, @obs:col, @obsm:X_pca.",
        ),
        click.option(
            "--compute-baseline",
            is_flag=True,
            default=False,
            help="If set, compute the task baseline from the (raw) representation.",
        ),
        click.option(
            "--random-seed",
            type=int,
            default=DEFAULT_SEED,
            show_default=True,
            help="Random seed.",
        ),
        click.option(
            "--output-file",
            type=click.Path(dir_okay=False, writable=True, resolve_path=True),
            help="If provided, write JSON results to this file.",
        ),
    ]


def add_task_parameter_option(parameter_name: str, param_info) -> click.Option:
    cli_flag = f"--{parameter_name.replace('_', '-')}"
    default_help = "" if param_info.required else f" (default: {param_info.default})"
    return click.option(
        cli_flag,
        parameter_name,
        required=param_info.required,
        type=str,
        help=f"{param_info.stringified_type}{default_help}",
    )


def add_baseline_parameter_option(parameter_name: str, param_info) -> click.Option:
    cli_flag = f"--baseline-{parameter_name.replace('_', '-')}"
    default_help = "" if param_info.required else f" (default: {param_info.default})"
    return click.option(
        cli_flag,
        f"baseline_{parameter_name}",
        required=False,
        type=str,
        help=f"[baseline] {param_info.stringified_type}{default_help}",
    )


@click.group(help="Run benchmark tasks.")
def run():
    pass


def add_dynamic_task_command(task_name: str):
    task_info = TASK_REGISTRY.get_task_info(task_name)
    command_help_text = TASK_REGISTRY.get_task_help(task_name)

    def task_execution_handler(**cli_kwargs):
        dataset_name: str = cli_kwargs.pop("dataset")
        cell_representation_path: Optional[str] = cli_kwargs.pop(
            "cell_representation_path"
        )
        should_compute_baseline: bool = cli_kwargs.pop("compute_baseline")
        random_seed: int = cli_kwargs.pop("random_seed")
        output_file_path: Optional[str] = cli_kwargs.pop("output_file")

        task_parameters: Dict[str, Any] = {}
        baseline_parameters: Dict[str, Any] = {}

        for param_name in task_info.task_params.keys():
            if param_name in cli_kwargs and cli_kwargs[param_name] is not None:
                task_parameters[param_name] = convert_cli_parameter(
                    cli_kwargs[param_name], task_info.task_params[param_name]
                )

        for param_name in task_info.baseline_params.keys():
            baseline_key = f"baseline_{param_name}"
            if baseline_key in cli_kwargs and cli_kwargs[baseline_key] is not None:
                baseline_parameters[param_name] = convert_cli_parameter(
                    cli_kwargs[baseline_key], task_info.baseline_params[param_name]
                )

        try:
            dataset = load_dataset(dataset_name)
            if not hasattr(dataset, "adata"):
                raise click.ClickException(
                    f"Dataset '{dataset_name}' does not provide an `.adata` attribute."
                )
            adata = dataset.adata

            if cell_representation_path is None:
                cell_representation = "@X"
            elif cell_representation_path.startswith("@"):
                cell_representation = cell_representation_path
            else:
                cell_representation = load_numpy_array_from_path(
                    cell_representation_path
                )

            execution_results = run_task(
                task_name=task_name,
                adata=adata,
                cell_representation=cell_representation,
                run_baseline=should_compute_baseline,
                baseline_params=baseline_parameters,
                task_params=task_parameters,
                random_seed=random_seed,
            )
        except Exception as execution_error:
            raise click.ClickException(str(execution_error)) from execution_error

        json_output = json.dumps(execution_results, indent=2, default=str)
        if output_file_path:
            with open(output_file_path, "w") as output_file:
                output_file.write(json_output)
        else:
            click.echo(json_output)

    task_command = click.command(name=task_name, help=command_help_text)(
        task_execution_handler
    )

    for param_name, param_info in reversed(list(task_info.baseline_params.items())):
        task_command = add_baseline_parameter_option(param_name, param_info)(
            task_command
        )

    for param_name, param_info in reversed(list(task_info.task_params.items())):
        task_command = add_task_parameter_option(param_name, param_info)(
            task_command
        )

    for cli_option in reversed(add_shared_cli_options()):
        task_command = cli_option(task_command)

    return task_command


for task_name in TASK_REGISTRY.list_tasks():
    run.add_command(add_dynamic_task_command(task_name))


__all__ = ["run"]
