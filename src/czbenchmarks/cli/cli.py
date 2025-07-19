import click
import pandas as pd
import sys
import logging

# CZ-Benchmarks imports
from czbenchmarks.datasets.utils import load_dataset
from .cli_list import get_resource_list
from .utils import (
    get_version,
    resolve_paths,
    parse_and_validate_params,
    prepare_task_input,
    execute_and_format_results,
    echo_results,
)
from .types import TASK_REGISTRY
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
    task_name = kwargs["task_name"]
    task_info = TASK_REGISTRY[task_name]

    if (
        task_info.get("requires_multiple_datasets", False)
        and len(kwargs["datasets"]) < 2
    ):
        raise click.UsageError(
            f"Task '{task_name}' requires at least two --dataset arguments."
        )

    # 1. Parse and validate all CLI parameters
    validated_params = parse_and_validate_params(task_name, kwargs)

    # 2. Load data and model outputs
    ds_objs = [load_dataset(ds) for ds in kwargs["datasets"]]
    model_outputs = resolve_paths(kwargs["embeddings"], "embedding", kwargs["datasets"])

    # 3. Prepare task-specific inputs
    pred_df = None
    if task_name == "perturbation":
        pred_df = pd.DataFrame(model_outputs[0], columns=ds_objs[0].adata.var_names)
        model_embedding = ds_objs[0].adata.X  # Control data is the input
    else:
        model_embedding = (
            model_outputs
            if task_info.get("requires_multiple_datasets")
            else model_outputs[0]
        )

    task_input = prepare_task_input(task_name, ds_objs, validated_params, pred_df)
    task_instance = task_info["task_class"](random_seed=kwargs["seed"])

    # 4. Execute and collect results
    all_results = execute_and_format_results(
        task_instance,
        task_name,
        model_embedding,
        task_input,
        "model",
        list(kwargs["datasets"]),
    )

    if kwargs["with_baseline"]:
        if task_name == "cross-species":
            # A standard PCA baseline on concatenated, unaligned raw counts from different
            # species is not a meaningful biological baseline, so it's skipped.
            log.warning(
                "Baseline computation is skipped for 'cross-species' as it is not biologically meaningful."
            )
        else:
            baseline_embedding = task_instance.compute_baseline(ds_objs[0].adata.X)
            all_results.extend(
                execute_and_format_results(
                    task_instance,
                    task_name,
                    baseline_embedding,
                    task_input,
                    "baseline",
                    list(kwargs["datasets"]),
                )
            )

    # 5. Output results
    echo_results(all_results, kwargs["output_format"], kwargs["output_file"])


def main():
    """Main entry point for the CLI."""
    cli(prog_name="czbenchmarks")


if __name__ == "__main__":
    main()
