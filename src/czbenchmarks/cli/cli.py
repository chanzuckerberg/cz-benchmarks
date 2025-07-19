import click
import sys
import logging

# CZ-Benchmarks imports
from .cli_list import get_resource_list
from .utils import (
    get_version,
)
from .types import TASK_REGISTRY
from .cli_run import run as run_task
# ----------------------------------------
# Setup and Constants
# ----------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
log = logging.getLogger(__name__)

VERSION = get_version()

# ----------------------------------------
# CLI Definition
# ----------------------------------------


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version=VERSION, prog_name="czbenchmarks")
def cli():
    """CZ-Benchmarks CLI: A tool for running standardized model evaluation tasks."""
    pass


@cli.command("list")
@click.argument("resource", type=click.Choice(["datasets", "tasks"]))
def list_resources(resource: str):
    """List available datasets or tasks."""
    get_resource_list(resource)


@cli.command("run")
@click.argument("task_name", type=click.Choice(list(TASK_REGISTRY.keys())))
@click.option(
    "-d",
    "--dataset",
    "datasets",
    multiple=True,
    required=True,
    help="Name of a dataset. Can be specified multiple times for multi-dataset tasks.",
)
@click.option(
    "-e",
    "--embedding",
    "embeddings",
    multiple=True,
    required=True,
    help="Path/glob to embedding file(s) (supports {dataset} formatting).",
)
@click.option(
    "--obs-key", help="Key in AnnData.obs for ground-truth labels (e.g., 'cell_type')."
)
@click.option("--batch-key", help="Key in AnnData.obs for batch information.")
@click.option(
    "-p",
    "--task-param",
    "task_params",
    multiple=True,
    help="Task-specific parameter, e.g., -p n_folds=5",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--with-baseline",
    is_flag=True,
    help="Also run task on a standard baseline (e.g., PCA).",
)
@click.option(
    "--output-format",
    default="json",
    show_default=True,
    type=click.Choice(["json", "yaml"]),
    help="Output format.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Write results to this file instead of stdout.",
)
def run(**kwargs):
    """
    Run a specified benchmark task on model output embeddings.

    TASK_NAME: The task to run (e.g., 'clustering'). Use 'czbenchmarks list tasks' for details.
    """
    run_task(**kwargs)


def main():
    """Main entry point for the CLI."""
    cli(prog_name="czbenchmarks")


if __name__ == "__main__":
    main()
