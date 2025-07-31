import click
import json
import numpy as np
import pandas as pd
from pydantic import ValidationError
from typing import Any, Callable, List
from ..datasets import load_dataset, Dataset
from ..tasks.types import CellRepresentation
from .types import TaskParameter, TaskDefinition
from ..metrics.utils import aggregate_results
import importlib.metadata
import subprocess
import functools
import logging
import tomli
from pathlib import Path
import inspect
from typing import Set


log = logging.getLogger(__name__)

_REPO_PATH = Path(__file__).parent.parent.parent.parent


def _get_pyproject_version() -> str:
    """
    Make an attempt to get the version from pyproject.toml
    """
    pyproject_path = _REPO_PATH / "pyproject.toml"

    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)
        return str(pyproject["project"]["version"])
    except Exception:
        log.exception("Could not determine cz-benchmarks version from pyproject.toml")

    return "unknown"


def _get_git_commit(base_version: str) -> str:
    """
    Return '' if the repo is exactly at the tag matching `base_version`
    (which should be what's in the pyproject file, with NO 'v' prepended)
    or '+<short-sha>[.dirty]' if not, where '.dirty' is added when there
    are uncommitted changes
    """
    if not (_REPO_PATH / ".git").exists():
        return ""

    tag = "v" + base_version  # this is our convention
    try:
        tag_commit = subprocess.check_output(
            ["git", "-C", str(_REPO_PATH), "rev-list", "-n", "1", tag],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        log.error("Could not find a commit hash for tag %r in git", tag)
        tag_commit = "error"

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(_REPO_PATH), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        log.error("Could not get current commit hash from git")
        commit = "unknown"

    try:
        is_dirty = (
            bool(  # the subprocess will return an empty string if the repo is clean
                subprocess.check_output(
                    ["git", "-C", str(_REPO_PATH), "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        )
    except subprocess.CalledProcessError:
        log.error("Could not get repo status from git")
        is_dirty = True

    if tag_commit == commit and not is_dirty:
        # if we're on the commit matching the version tag, then our version is simply the tag
        return ""
    else:
        # otherwise we want to add the commit hash and dirty status
        dirty_string = ".dirty" if is_dirty else ""
        return f"+{commit[:7]}{dirty_string}"


@functools.cache
def get_version() -> str:
    """
    Get the current version of the cz-benchmarks library
    """
    try:
        version = importlib.metadata.version("cz-benchmarks")  # yes, with the hyphen
    except importlib.metadata.PackageNotFoundError:
        log.debug(
            "Package `cz-benchmarks` is not installed: fetching version info from pyproject.toml"
        )
        version = _get_pyproject_version()

    git_commit = _get_git_commit(version)
    return "v" + version + git_commit


# These names are sourced from data objects, not the CLI.
RESERVED_INPUT_NAMES: Set[str] = {
    "obs",
    "var_names",
    "input_labels",
    "labels",
    "batch_labels",
    "perturbation_truth",
    "organism_list",
    "cell_representation",
}
RESERVED_BASELINE_NAMES: Set[str] = {
    "self",
    "expression_data",
    "cell_representation",
    "kwargs",
    "obs_names",
    "var_names",
}


def add_options(options: List[TaskParameter]) -> Callable:
    """A decorator that adds a list of click options to a command."""

    def decorator(f):
        for option in reversed(options):
            f = click.option(
                f"--{option.name.replace('_', '-')}",
                option.name,
                type=option.type,
                help=option.help,
                default=option.default,
                required=option.required,
                is_flag=option.is_flag,
            )(f)
        return f

    return decorator


def discover_task_parameters(task_def: TaskDefinition) -> List[TaskParameter]:
    """Inspects a Task's Input model and discovers its CLI parameters."""
    params = []
    for name, field in task_def.input_model.model_fields.items():
        if name in RESERVED_INPUT_NAMES:
            continue
        params.append(
            TaskParameter(
                name=name,
                type=field.annotation,
                help=field.description or "",
                default=field.default,
                required=field.is_required(),
            )
        )
    return params


def discover_baseline_parameters(task_def: TaskDefinition) -> List[TaskParameter]:
    """Inspects a Task's compute_baseline method for configurable args."""
    params = []
    try:
        sig = inspect.signature(task_def.task_class.compute_baseline)
        for param in sig.parameters.values():
            if param.name in RESERVED_BASELINE_NAMES or param.kind == param.VAR_KEYWORD:
                continue
            params.append(
                TaskParameter(
                    name=f"baseline_{param.name}",
                    type=param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else str,
                    help=f"Baseline parameter for {param.name}.",
                    default=param.default
                    if param.default != inspect.Parameter.empty
                    else None,
                    required=param.default == inspect.Parameter.empty,
                )
            )
    except (TypeError, AttributeError):
        pass  # Task may not have a compute_baseline method
    return params


def get_datasets(dataset_names: List[str]) -> List[Dataset]:
    """Loads a list of datasets by name."""
    try:
        return [load_dataset(name) for name in dataset_names]
    except Exception as e:
        raise click.UsageError(f"Failed to load dataset: {e}")


def load_embedding(path: str) -> CellRepresentation:
    """Loads a model embedding from a file."""
    try:
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path, index_col=0).values
        else:
            raise NotImplementedError(
                "Only .npy and .csv embedding files are supported."
            )
    except Exception as e:
        raise click.BadParameter(f"Could not load embedding from '{path}': {e}")


def prepare_task_inputs(
    task_def: TaskDefinition, datasets: List[Dataset], cli_args: dict
) -> Any:
    """Gathers data from all sources, validates, and constructs the Pydantic TaskInput model."""
    available_data = {}

    if task_def.requires_multiple_datasets:
        available_data["organism_list"] = [d.organism for d in datasets]
        available_data["labels"] = (
            [d.adata.obs[cli_args["label_key"]] for d in datasets]
            if "label_key" in cli_args
            else []
        )
    else:
        dataset = datasets[0]
        available_data["obs"] = dataset.adata.obs
        available_data["var_names"] = dataset.adata.var_names
        available_data["perturbation_truth"] = getattr(
            dataset, "perturbation_truth", None
        )

        if cli_args.get("label_key"):
            label_key = cli_args["label_key"]
            if label_key not in dataset.adata.obs:
                raise click.BadParameter(
                    f"Label key '{label_key}' not found in dataset.",
                    param_hint="--label-key",
                )
            available_data["input_labels"] = dataset.adata.obs[label_key]
            available_data["labels"] = dataset.adata.obs[label_key]

        if cli_args.get("batch_key"):
            batch_key = cli_args["batch_key"]
            if batch_key not in dataset.adata.obs:
                raise click.BadParameter(
                    f"Batch key '{batch_key}' not found in dataset.",
                    param_hint="--batch-key",
                )
            available_data["batch_labels"] = dataset.adata.obs[batch_key]

    available_data.update(cli_args)

    model_fields = task_def.input_model.model_fields.keys()
    final_args = {k: v for k, v in available_data.items() if k in model_fields}

    try:
        return task_def.input_model(**final_args)
    except ValidationError as e:
        error_msg = f"Invalid parameters for task '{task_def.display_name}':\n"
        for error in e.errors():
            loc = " -> ".join(map(str, error["loc"]))
            error_msg += f"  - {loc}: {error['msg']}\n"
        raise click.UsageError(error_msg)


def write_results(results: List[Any], output_file: str | None):
    """Aggregates metrics and writes results to stdout or a file in JSON format."""
    aggregated = aggregate_results(results)
    results_dict = [res.model_dump(mode="json") for res in aggregated]

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo("\n--- RESULTS ---")
        click.echo(json.dumps(results_dict, indent=2))
