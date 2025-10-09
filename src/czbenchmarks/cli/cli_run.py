from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, get_args

import click
import numpy as np
import pandas as pd

from czbenchmarks.constants import RANDOM_SEED as DEFAULT_SEED
from czbenchmarks.datasets import load_dataset
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.utils import load_local_dataset
from czbenchmarks.tasks.task import TASK_REGISTRY, TaskParameter

from .runner import run_task

logger = logging.getLogger(__name__)


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


def _to_organism_enum(s: str) -> Organism:
    """Convert a string to an Organism enum, case-insensitive.

    Supports enum names (e.g. "HUMAN"), species names (e.g. "homo_sapiens"),
    gene prefix (e.g. "ENSG"), and common names like "human" or "mouse".

    Args:
        s: String to convert to Organism enum

    Returns:
        Organism enum value

    Raises:
        ValueError: If the string cannot be converted to an Organism
    """
    user_organism = s.strip().lower()
    for organism in Organism:
        logger.debug(
            f"Resolving organism from: {user_organism}, to: {organism._name_.lower()}, "
            f"{organism.value[0].lower()}, {organism.value[1].lower()}"
        )
        # Match enum name (e.g. "HUMAN")
        if user_organism == organism._name_.lower():
            return organism
        # Match species name (e.g. "homo_sapiens")
        if user_organism == organism.value[0].lower():
            return organism
        # Match gene prefix (e.g. "ENSG")
        if organism.value[1] and user_organism == organism.value[1].lower():
            return organism

    valid_names = ", ".join([org._name_ for org in Organism])
    logger.error(
        f"Cannot convert '{s}' to Organism enum. Valid values are: {valid_names}"
    )
    raise ValueError(
        f"Cannot convert '{s}' to Organism enum. Valid values are: {valid_names}"
    )


def _align_labels_to_organisms(
    labels: List[str], organisms: List[str], default_label: str = "cell_type"
) -> Tuple[List[str], List[Organism]]:
    """Align and process labels and organisms for cross-species tasks.

    - Ensures each organism has a corresponding label
    - Converts label names to AnnData reference format if needed
    - Pads labels with default if fewer labels than organisms
    - Parses organism strings into Organism enums

    Args:
        labels: List of label strings (may be column names or references)
        organisms: List of organism strings (e.g., "homo_sapiens", "HUMAN", etc.)
        default_label: Default label to use if not enough labels provided

    Returns:
        Tuple of (processed_labels, organisms_list)
            processed_labels: List of label references, one per organism
            organisms_list: List of Organism enums
    """
    num_organisms = len(organisms)
    labels = list(labels) if labels else []

    # Pad labels if fewer than organisms
    if len(labels) < num_organisms:
        labels += [default_label] * (num_organisms - len(labels))
    # Truncate if more labels than organisms
    if len(labels) > num_organisms:
        labels = labels[:num_organisms]

    processed_labels = []
    for idx, label in enumerate(labels):
        if isinstance(label, str) and label.startswith("@"):
            processed_labels.append(label)
        else:
            # For multi-dataset, use indexed references
            processed_labels.append(f"@{idx}:obs:{label}")

    organisms_list = []
    for organism_str in organisms:
        # Remove any suffix after colon if present
        org_name = organism_str.split(":", 1)[0]
        org_enum = _to_organism_enum(org_name)
        organisms_list.append(org_enum)

    return processed_labels, organisms_list


def _normalize_label_param(name: str, value: Any) -> Any:
    """Normalize common label-like parameters to AnnData references.

    Converts simple column names like 'cell_type' to '@obs:cell_type' format.
    Handles both single values and lists of values.

    Args:
        name: Parameter name
        value: Parameter value to normalize

    Returns:
        Normalized value with proper AnnData reference format
    """
    # Parameters that should be normalized to @obs:column format
    label_like = {"labels", "input_labels", "batch_labels", "sample_ids"}

    if name not in label_like or not value:
        return value

    # Handle single string value
    if isinstance(value, str):
        if not value.startswith("@"):
            return f"@obs:{value}"
        return value

    # Handle list/tuple of values
    if isinstance(value, (list, tuple)):
        normalized = []
        for v in value:
            if isinstance(v, str) and not v.startswith("@"):
                normalized.append(f"@obs:{v}")
            else:
                normalized.append(v)
        return normalized

    return value


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

    # Determine click type from Python type
    click_type = click.STRING
    if param.type is bool:
        click_type = click.BOOL
    elif param.type is int:
        click_type = click.INT
    elif param.type is float:
        click_type = click.FLOAT

    # Handle Literal types (choices)
    param_str = str(param.type)
    if "Literal" in param_str:
        try:
            choices = get_args(param.type)
            if choices:
                click_type = click.Choice([str(c) for c in choices])
        except Exception:
            pass

    # Build help text
    help_text = param.help_text
    if not param.required and param.default is not None:
        help_text += f" (Default: {param.default})"
    if param.is_multiple:
        help_text += " [multiple]"

    # Build option kwargs
    kwargs = {
        "type": click_type,
        "multiple": param.is_multiple,
        "default": param.default if not param.is_multiple else None,
        "help": help_text,
        "required": param.required and param.default is None,
    }

    # Special handling for boolean flags
    if click_type is click.BOOL:
        kwargs["is_flag"] = True
        kwargs["default"] = bool(param.default) if param.default is not None else False
        return click.Option([long_name, f"--no-{long_name.lstrip('-')}"], **kwargs)

    return click.Option([long_name], **kwargs)


def _extract_and_normalize_cli_kwargs(
    task_key: str, cli_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Extract and normalize CLI parameters into task and baseline dictionaries.

    Separates parameters by task vs baseline, normalizes label-like parameters,
    and handles special cases like organism resolution.

    Args:
        task_key: The task identifier
        cli_params: Raw CLI parameters from Click

    Returns:
        Tuple of (task_kwargs, baseline_kwargs, baseline_param_seen)
            task_kwargs: Normalized task parameters
            baseline_kwargs: Normalized baseline parameters
            baseline_param_seen: Whether any baseline param was provided
    """
    info = TASK_REGISTRY.get_task_info(task_key)
    task_kwargs = {}
    baseline_kwargs = {}
    baseline_param_seen = False

    # Extract task parameters
    # Note: Click converts --kebab-case to snake_case in params dict
    for name in info.task_params:
        if name in cli_params and cli_params[name] is not None:
            val = cli_params[name]
            # Skip empty tuples from multiple params
            if isinstance(val, tuple) and not val:
                continue
            # Normalize label-like parameters
            task_kwargs[name] = _normalize_label_param(name, val)

    # Extract baseline parameters
    for name in info.baseline_params:
        baseline_key = f"baseline_{name}"
        if baseline_key in cli_params and cli_params[baseline_key] is not None:
            val = cli_params[baseline_key]
            if isinstance(val, tuple) and not val:
                continue
            baseline_kwargs[name] = val
            baseline_param_seen = True

    # Special handling for multi-dataset tasks with organisms
    if info.requires_multiple_datasets:
        if "organisms" in task_kwargs or "organism_list" in task_kwargs:
            org_key = "organisms" if "organisms" in task_kwargs else "organism_list"
            organisms = task_kwargs[org_key]

            # Get labels if present
            label_key = None
            for key in ["labels", "input_labels"]:
                if key in task_kwargs:
                    label_key = key
                    break

            labels = task_kwargs.get(label_key, []) if label_key else []

            try:
                aligned_labels, organism_enums = _align_labels_to_organisms(
                    labels, organisms
                )
                if label_key:
                    task_kwargs[label_key] = aligned_labels
                task_kwargs[org_key] = organism_enums
            except ValueError as e:
                raise click.ClickException(f"Organism resolution failed: {e}")

    return task_kwargs, baseline_kwargs, baseline_param_seen


def convert_cli_parameter(param_value: str, param_info) -> Any:
    """Convert CLI parameter string to appropriate Python type.

    Args:
        param_value: String value from CLI
        param_info: TaskParameter with type information

    Returns:
        Converted value in appropriate Python type
    """
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
    def mutually_exclusive_callback(ctx, param, value):
        if value is None:
            return
        other_option = "dataset" if param.name == "user_dataset" else "user_dataset"
        if ctx.params.get(other_option) is not None:
            raise click.UsageError(
                f"'{param.opts[0]}' and '--{other_option}' are mutually exclusive."
            )
        return value

    return [
        click.option(
            "-d",
            "--dataset",
            help="Dataset name available in czbenchmarks.",
            callback=mutually_exclusive_callback,
        ),
        click.option(
            "-u",
            "--user-dataset",
            help='Path to a user .h5ad file as JSON: \'{"dataset_class": "SingleCellDataset", "organism": "Organism.Human", "path": "~/mydata.h5ad"}\'.',
            callback=mutually_exclusive_callback,
        ),
        click.option(
            "-c",
            "--cell-representation-path",
            help="Path to embedding array (.npy/.npz/.csv/.tsv) or AnnData reference like @X, @obsm:X_pca.",
            default="@X",
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
        # Extract shared CLI options
        dataset_name: str = cli_kwargs.pop("dataset", None)
        user_dataset_json: Optional[str] = cli_kwargs.pop("user_dataset", None)
        cell_representation_path: Optional[str] = cli_kwargs.pop(
            "cell_representation_path", "@X"
        )
        should_compute_baseline: bool = cli_kwargs.pop("compute_baseline", False)
        random_seed: int = cli_kwargs.pop("random_seed", DEFAULT_SEED)
        output_file_path: Optional[str] = cli_kwargs.pop("output_file", None)

        # Extract and normalize task-specific parameters
        try:
            task_parameters, baseline_parameters, baseline_seen = (
                _extract_and_normalize_cli_kwargs(task_name, cli_kwargs)
            )

            # If user explicitly requested baseline or provided baseline params
            if should_compute_baseline or baseline_seen:
                should_compute_baseline = True

            # Note: Validation happens in runner.py after AnnData references are resolved

        except Exception as e:
            raise click.ClickException(f"Error processing parameters: {e}")

        # Load dataset
        try:
            if user_dataset_json:
                user_dataset = json.loads(user_dataset_json)
                required_keys = ["dataset_class", "organism", "path"]
                missing_keys = [k for k in required_keys if k not in user_dataset]
                if missing_keys:
                    raise click.ClickException(
                        f"Missing required key(s) in --user-dataset JSON: {', '.join(missing_keys)}. "
                        'Example: \'{"dataset_class": "czbenchmarks.datasets.Dataset", "organism": "HUMAN", "path": "~/mydata.h5ad"}\''
                    )
                resolved_path = os.path.expanduser(user_dataset["path"])
                if not os.path.exists(resolved_path):
                    raise click.ClickException(
                        f"The file specified in --user-dataset 'path' does not exist: {resolved_path}"
                    )
                dataset = load_local_dataset(
                    dataset_class=user_dataset["dataset_class"],
                    organism=user_dataset["organism"],
                    path=str(resolved_path),
                )
            elif dataset_name:
                dataset = load_dataset(dataset_name)
            else:
                raise click.ClickException(
                    "You must specify either --dataset or --user-dataset."
                )

            if not hasattr(dataset, "adata"):
                raise click.ClickException(
                    "Dataset does not provide an `.adata` attribute."
                )
            adata = dataset.adata

        except Exception as dataset_error:
            raise click.ClickException(f"Error loading dataset: {dataset_error}")

        # Load cell representation
        try:
            if cell_representation_path.startswith("@"):
                cell_representation = cell_representation_path
            else:
                cell_representation = load_numpy_array_from_path(
                    cell_representation_path
                )
        except Exception as rep_error:
            raise click.ClickException(
                f"Error loading cell representation: {rep_error}"
            )

        # Execute task
        try:
            logger.info(f"Executing task '{task_name}'...")
            execution_results = run_task(
                task_name=task_name,
                adata=adata,
                cell_representation=cell_representation,
                run_baseline=should_compute_baseline,
                baseline_params=baseline_parameters if should_compute_baseline else {},
                task_params=task_parameters,
                random_seed=random_seed,
            )
        except Exception as execution_error:
            raise click.ClickException(f"Task execution failed: {execution_error}")

        # Output results
        json_output = json.dumps(execution_results, indent=2, default=str)
        if output_file_path:
            with open(output_file_path, "w") as output_file:
                output_file.write(json_output)
            logger.info(f"Results written to {output_file_path}")
        else:
            click.echo(json_output)

    # Create the command with enhanced metadata
    task_command = click.command(
        name=task_name,
        help=task_info.description,
        context_settings={"help_option_names": ["-h", "--help"]},
    )(task_execution_handler)

    # Add baseline parameters first (they appear last in help text)
    for param_info in reversed(list(task_info.baseline_params.values())):
        option = _build_option_from_param(param_info, is_baseline=True)
        task_command.params.append(option)

    # Add task-specific parameters
    for param_info in reversed(list(task_info.task_params.values())):
        option = _build_option_from_param(param_info, is_baseline=False)
        task_command.params.append(option)

    # Add shared CLI options
    for cli_option in reversed(add_shared_cli_options()):
        task_command = cli_option(task_command)

    return task_command


for task_name in TASK_REGISTRY.list_tasks():
    run.add_command(add_dynamic_task_command(task_name))


__all__ = ["run"]
