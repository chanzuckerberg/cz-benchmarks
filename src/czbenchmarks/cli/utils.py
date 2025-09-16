import functools
import importlib.metadata
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, List

import click
import numpy as np
import pandas as pd
import tomli

from ..datasets import Dataset, load_dataset
from ..metrics.utils import aggregate_results
from ..tasks.types import CellRepresentation

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


def mutually_exclusive(*options):
    def _callback(ctx, param, value):
        if value is not None:
            for other in options:
                if ctx.params.get(other) is not None:
                    raise click.UsageError(
                        f"Options --{param.name.replace('_', '-')} and --{other.replace('_', '-')} are mutually exclusive."
                    )
        return value

    return _callback
