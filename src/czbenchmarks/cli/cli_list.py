import argparse
import sys

from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.models import utils as model_utils
from czbenchmarks.cli import cli


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add list command arguments to the parser.
    """
    parser.add_argument(
        "list_type",
        choices=["datasets", "models", "tasks"],
        help="List available datasets, models, or tasks.",
    )


def main(args: argparse.Namespace) -> None:
    """
    List available datasets, models, or tasks.
    """
    if args.list_type == "datasets":
        sys.stdout.write(" ".join(dataset_utils.list_available_datasets()))
    elif args.list_type == "models":
        sys.stdout.write(" ".join(model_utils.list_available_models()))
    elif args.list_type == "tasks":
        sys.stdout.write(" ".join(cli.TASKS))
    sys.stdout.write("\n")
