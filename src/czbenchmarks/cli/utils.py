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
import os
from pathlib import Path

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

def add_options(options: List[TaskParameter], is_baseline: bool = False) -> Callable:
    """A decorator that adds a list of click options to a command."""
    def decorator(f):
        for option in reversed(options):
            # Prefix baseline parameters to avoid clashes
            param_name = f"--baseline-{option.name.replace('_', '-')}" if is_baseline else f"--{option.name.replace('_', '-')}"
            kwarg_name = f"baseline_{option.name}" if is_baseline else option.name

            f = click.option(
                param_name,
                kwarg_name,
                type=option.type,
                help=f"{option.help}{' (for baseline)' if is_baseline else ''}",
                default=option.default,
                required=option.required,
                is_flag=option.is_flag,
            )(f)
        return f
    return decorator


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
            return pd.read_csv(path).values
        else:
            raise NotImplementedError("Only .npy and .csv embedding files are supported.")
    except Exception as e:
        raise click.BadParameter(f"Could not load embedding from '{path}': {e}")


def prepare_task_inputs(task_def: TaskDefinition, datasets: List[Dataset], cli_args: dict) -> Any:
    """
    Gathers all necessary data from datasets and CLI args, then validates
    and constructs the Pydantic TaskInput model.
    """
    # This dictionary holds everything the TaskInput model might need.
    # It sources data from the loaded datasets first, then from CLI arguments.
    available_data = {}
    
    # Handle single vs. multi-dataset tasks
    if task_def.requires_multiple_datasets:
        available_data["organism_list"] = [d.organism for d in datasets]
        available_data["labels"] = [getattr(d, 'labels', None) for d in datasets]
    else:
        # Most tasks operate on a single dataset
        dataset = datasets[0]
        available_data["obs"] = dataset.adata.obs
        available_data["var_names"] = dataset.adata.var_names
        # Use getattr to safely access optional attributes
        available_data["input_labels"] = getattr(dataset, 'labels', None)
        available_data["labels"] = getattr(dataset, 'labels', None)
        available_data["perturbation_truth"] = getattr(dataset, 'perturbation_truth', None)
        if "batch_key" in cli_args and cli_args["batch_key"]:
            available_data["batch_labels"] = dataset.adata.obs[cli_args["batch_key"]]

    # Add remaining CLI args
    available_data.update(cli_args)
    
    # Filter available_data to only what the model needs
    model_fields = task_def.input_model.model_fields.keys()
    final_args = {k: v for k, v in available_data.items() if k in model_fields}

    try:
        return task_def.input_model(**final_args)
    except ValidationError as e:
        # Provide clean, user-friendly error messages
        error_msg = f"Invalid parameters for task '{task_def.display_name}':\n"
        for error in e.errors():
            loc = " -> ".join(map(str, error['loc']))
            error_msg += f"  - {loc}: {error['msg']}\n"
        raise click.UsageError(error_msg)


def write_results(results: List[Any], output_file: str | None):
    """Aggregates metrics and writes results to stdout or a file in JSON format."""
    aggregated = aggregate_results(results)
    results_dict = [res.model_dump(mode="json") for res in aggregated]

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        click.echo(f"Results saved to {output_file}")
    else:
        click.echo("\n--- RESULTS ---")
        click.echo(json.dumps(results_dict, indent=2))