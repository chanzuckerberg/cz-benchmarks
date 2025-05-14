"""
czbenchmarks CLI

Usage:
    czbenchmarks list [datasets|models|tasks]
    czbenchmarks run --models <model_name> --datasets <dataset_name> --tasks <task_name>
        [--output-file <output_file>]
        [--save-processed-datasets <output_dir>]
"""

import argparse
import logging
import sys
from czbenchmarks.cli import cli_list, cli_run
from czbenchmarks.cli.utils import get_version

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


if __name__ == "__main__":
    main()
