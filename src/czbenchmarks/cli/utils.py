import collections
import functools
import importlib.metadata
import itertools
import logging
from pathlib import Path
import subprocess
import typing

import tomli

from czbenchmarks.cli.types import TaskResult
import czbenchmarks.metrics.utils as metric_utils


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


def aggregate_task_results(results: typing.Iterable[TaskResult]) -> list[TaskResult]:
    """Aggregate the task results by task_name, model (with args), and set(datasets).
    Each new result will have a new set of metrics, created by aggregating together
    metrics of the same type.
    """
    grouped_results = collections.defaultdict(list)
    for result in results:
        grouped_results[result.aggregation_key].append(result)

    aggregated = []
    for results_to_agg in grouped_results.values():
        aggregated_metrics = metric_utils.aggregate_results(
            list(
                itertools.chain.from_iterable(tr.metrics for tr in results_to_agg)
            )  # cast to list is unnecessary but helps testing
        )
        if any(tr.runtime_metrics for tr in results_to_agg):
            raise ValueError(
                "Aggregating runtime_metrics for TaskResults is not supported"
            )

        first_result = results_to_agg[0]  # everything but the metrics should be common
        aggregated_result = TaskResult(
            task_name=first_result.task_name,
            task_name_display=first_result.task_name_display,
            # model=first_result.model,
            datasets=first_result.datasets,
            metrics=aggregated_metrics,
            runtime_metrics={},
        )
        aggregated.append(aggregated_result)
    return aggregated
