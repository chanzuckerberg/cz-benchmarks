import functools
import importlib.metadata
import logging
import subprocess
from pathlib import Path

import tomli


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
