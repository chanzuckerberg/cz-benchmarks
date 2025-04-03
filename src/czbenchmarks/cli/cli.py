"""
czbenchmarks CLI

Usage:
    czbenchmarks list [datasets|models|tasks]
    czbenchmarks run --models <model_name> --datasets <dataset_name> --tasks <task_name> --label-key <label_key>
        [--output-file <output_file>]
        [--save-processed-datasets <output_dir>]
"""

import argparse
import logging
import sys
import tomli
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from czbenchmarks.cli import cli_list, cli_run

log = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the czbenchmarks CLI."""

    parser = argparse.ArgumentParser(
        description="czbenchmark: A command-line utility for single-cell benchmark tasks."
    )
    parser.add_argument(
        "--version",
        help="Show version number and exit",
        action="version",
        version=f"%(prog)s {get_version()}",
    )
    parser.add_argument(
        "--log-level",
        "-ll",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level (default is info)",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a set of tasks.")
    list_parser = subparsers.add_parser("list", help="List datasets, models, or tasks.")

    cli_run.add_arguments(run_parser)
    cli_list.add_arguments(list_parser)

    # Parse arguments to dict
    try:
        args = parser.parse_args()
        logging.basicConfig(level=args.log_level.upper(), stream=sys.stdout)
    except argparse.ArgumentError as e:
        parser.error(str(e))
    except SystemExit:
        raise

    if args.action == "list":
        cli_list.main(args)

    elif args.action == "run":
        cli_run.main(args)


def get_version() -> str:
    """
    Get the current version of the czbenchmarks library.
    """
    try:
        return version("czbenchmarks")
    except PackageNotFoundError:
        log.debug(
            "Package `czbenchmarks` is not installed: fetching version info from pyproject.toml"
        )

    # In development this lib might not be installed as a package so try loading from pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


if __name__ == "__main__":
    main()
