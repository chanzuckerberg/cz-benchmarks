import collections
import functools
import importlib.metadata
import itertools
import logging
from pathlib import Path
import subprocess
import typing

import tomli

# from czbenchmarks.cli.types import TaskResult
import czbenchmarks.metrics.utils as metric_utils

import click
import yaml
import json
import glob
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

# CZ-Benchmarks imports
from czbenchmarks.datasets.dataset import Dataset
from czbenchmarks.tasks.task import Task, TaskInput
from .types import TASK_REGISTRY

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
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


# def aggregate_task_results(results: typing.Iterable[TaskResult]) -> list[TaskResult]:
#     """Aggregate the task results by task_name, model (with args), and set(datasets).
#     Each new result will have a new set of metrics, created by aggregating together
#     metrics of the same type.
#     """
#     grouped_results = collections.defaultdict(list)
#     for result in results:
#         grouped_results[result.aggregation_key].append(result)

#     aggregated = []
#     for results_to_agg in grouped_results.values():
#         aggregated_metrics = metric_utils.aggregate_results(
#             list(
#                 itertools.chain.from_iterable(tr.metrics for tr in results_to_agg)
#             )  # cast to list is unnecessary but helps testing
#         )
#         if any(tr.runtime_metrics for tr in results_to_agg):
#             raise ValueError(
#                 "Aggregating runtime_metrics for TaskResults is not supported"
#             )

#         first_result = results_to_agg[0]  # everything but the metrics should be common
#         aggregated_result = TaskResult(
#             task_name=first_result.task_name,
#             task_name_display=first_result.task_name_display,
#             # model=first_result.model,
#             datasets=first_result.datasets,
#             metrics=aggregated_metrics,
#             runtime_metrics={},
#         )
#         aggregated.append(aggregated_result)
#     return aggregated


# ----------------------------------------
# Helper Functions
# ----------------------------------------


def echo_results(results: List[Dict[str, Any]], fmt: str, out_path: str) -> None:
    """Wraps metrics into the standard output structure and prints or writes it."""
    payload = {
        "czbenchmarks_version": get_version(),
        "args": "czbenchmarks " + " ".join(sys.argv[1:]),
        "task_results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    text = (
        yaml.dump(payload, sort_keys=False)
        if fmt == "yaml"
        else json.dumps(payload, indent=2, default=str)
    )
    if out_path:
        Path(out_path).write_text(text)
        log.info(f"Results written to {out_path}")
    else:
        click.echo(text)


def resolve_paths(
    patterns: Tuple[str, ...], friendly_name: str, ds_names: List[str]
) -> List[np.ndarray]:
    """
    Resolves file globs to numpy arrays, matching against dataset names.
    Includes basic validation and performance warnings.
    """
    if len(patterns) != len(ds_names):
        raise click.UsageError(
            f"Count mismatch: {len(patterns)} {friendly_name}(s) vs {len(ds_names)} dataset(s)."
        )

    loaded_data = []
    for pattern, ds_name in zip(patterns, ds_names):
        path_str = pattern.format(dataset=ds_name)
        matches = glob.glob(path_str)
        if not matches:
            raise click.UsageError(
                f"No file found for {friendly_name} pattern: '{path_str}'"
            )

        file_path = Path(matches[0])
        if len(matches) > 1:
            log.warning(
                f"Multiple files found for '{path_str}'. Using first: {file_path}"
            )

        # Warn user about potentially large files
        if file_path.stat().st_size > 4 * 1024**3:  # 4 GB
            log.warning(
                f"File '{file_path}' is large and may consume significant memory."
            )

        if file_path.suffix == ".npy":
            loaded_data.append(np.load(file_path))
        elif file_path.suffix == ".csv":
            loaded_data.append(pd.read_csv(file_path).values)
        else:
            raise click.UsageError(
                f"Unsupported file type '{file_path.suffix}'. Use .npy or .csv."
            )

    return loaded_data


def parse_and_validate_params(
    task_name: str, raw_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parses, validates, and type-casts all CLI parameters against the TASK_REGISTRY.

    Args:
        task_name: The name of the task being run.
        raw_params: A dictionary of all relevant CLI options.

    Returns:
        A cleaned dictionary of validated parameters for the task.
    """
    task_info = TASK_REGISTRY[task_name]
    valid_param_defs = task_info["params"]
    parsed_params = {}

    # Process special keys like --obs-key and --batch-key
    for key, source in [("obs_key", "obs_key"), ("batch_key", "batch_key")]:
        if raw_params.get(source):
            for param_name, param_source in valid_param_defs.items():
                if param_source == source:
                    parsed_params[param_name] = raw_params[source]

    # Process generic --task-param KEY=VALUE pairs
    for param_str in raw_params.get("task_params", []):
        if "=" not in param_str:
            raise click.UsageError(
                f"Invalid task parameter '{param_str}'. Must be in KEY=VALUE format."
            )
        key, value = param_str.split("=", 1)

        if key not in valid_param_defs:
            raise click.UsageError(
                f"Invalid parameter '{key}' for task '{task_name}'. Valid options are: {list(valid_param_defs.keys())}"
            )

        # Type-cast the value based on registry definition
        expected_type = valid_param_defs[key]
        if isinstance(expected_type, type):
            try:
                parsed_params[key] = expected_type(value)
            except ValueError:
                raise click.UsageError(
                    f"Cannot cast value for '{key}' to {expected_type.__name__}."
                )
        else:
            parsed_params[key] = value

    return parsed_params


def prepare_task_input(
    task_name: str,
    ds_objs: List[Dataset],
    validated_params: Dict[str, Any],
    pred_df: pd.DataFrame = None,
) -> TaskInput:
    """
    Constructs the appropriate TaskInput pydantic model.
    Assumes parameters have already been parsed and validated.
    """
    task_info = TASK_REGISTRY[task_name]
    input_class = task_info["input_class"]
    input_params = validated_params.copy()

    # Populate data from dataset objects based on validated keys
    for key, value in validated_params.items():
        if (
            value in [d.adata.obs.name for d in ds_objs]
            or value in ds_objs[0].adata.obs.columns
        ):
            if task_info.get("requires_multiple_datasets"):
                input_params[key] = [d.adata.obs[value] for d in ds_objs]
            else:
                input_params[key] = ds_objs[0].adata.obs[value]

    # Add special data for specific tasks
    if task_name == "perturbation":
        truth_data = ds_objs[0].perturbation_truth
        input_params.update(
            {
                "var_names": list(ds_objs[0].adata.var_names),
                "perturbation_truth": truth_data,
                "perturbation_pred": pred_df,
            }
        )
        if "gene_pert" not in input_params:
            default_gene = next(iter(truth_data.keys()))
            log.info(
                f"Defaulting to first available perturbation gene: '{default_gene}'"
            )
            input_params["gene_pert"] = default_gene

    if task_name == "cross-species":
        input_params["organism_list"] = [d.organism for d in ds_objs]

    return input_class(**input_params)


def execute_and_format_results(
    task: Task,
    task_name: str,
    embedding: Any,
    task_input: TaskInput,
    mode: str,
    ds_names: List[str],
) -> List[Dict[str, Any]]:
    """Runs a task and formats the MetricResult objects into dictionaries."""
    results = []
    for mr in task.run(embedding, task_input):
        result_entry = {
            "dataset": ",".join(ds_names),
            "task": task_name,
            "mode": mode,
            "metric": mr.metric_type.value,
            "value": mr.value,
        }
        if mr.params:
            result_entry["params"] = mr.params
        results.append(result_entry)
    return results
