import argparse
import sys


from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.tasks import utils as task_utils


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add list command arguments to the parser.
    """
    parser.add_argument(
        "list_type",
        choices=["datasets", "tasks"],
        help="List available datasets or tasks.",
    )


def get_resource_list(resource: str) -> str:
    """
    List available datasets or tasks.
    """
    if resource == "datasets":
        sys.stdout.write(" ".join(dataset_utils.list_available_datasets()))
    elif resource == "tasks":
        sys.stdout.write(" ".join(task_utils.TASK_NAMES))
    sys.stdout.write("\n")


def main(args: argparse.Namespace) -> None:
    get_resource_list(args.list_type)
