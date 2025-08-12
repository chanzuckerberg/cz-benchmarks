import click
import logging
import typing
import numpy as np
import pandas as pd

from ..tasks.task import TASK_REGISTRY
from ..tasks.runner import run_task
from .utils import get_datasets, write_results
from ..metrics.types import MetricResult

log = logging.getLogger(__name__)

# --- NEW: Utility functions for file loading and type casting, local to the CLI ---


def load_from_path(path: str) -> typing.Any:
    """Loads data from a file path based on its extension."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith((".txt", ".list")):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    raise click.BadParameter(f"Unsupported file type for path: {path}")


def cast_to_type(value: str, target_type: typing.Type) -> typing.Any:
    """Casts a string value to a given Python type."""
    try:
        if target_type is bool:
            return value.lower() in ["true", "1", "yes"]
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        return value
    except (ValueError, TypeError):
        return value


# --- Main CLI Group ---


@click.group(invoke_without_command=True)
@click.pass_context
def run(ctx):
    """Run a benchmark task."""
    if ctx.invoked_subcommand is None:
        click.echo("Please specify a task to run, e.g., 'czbenchmarks run clustering'.")
        click.echo(f"Available tasks: {', '.join(TASK_REGISTRY.list_tasks())}")


# --- Dynamic Command Creation ---

for task_name in TASK_REGISTRY.list_tasks():
    task_info = TASK_REGISTRY.get_task_info(task_name)

    @click.command(name=task_name, help=task_info.description)
    def task_callback(**kwargs):
        ctx = click.get_current_context()
        invoked_task_name = ctx.command.name
        log.info(f"Starting task: '{invoked_task_name}'")

        try:
            # 1. Separate CLI arguments into logical groups
            invoked_task_info = TASK_REGISTRY.get_task_info(invoked_task_name)
            task_arg_keys = set(invoked_task_info.task_params.keys())

            cli_args = {
                k: v
                for k, v in kwargs.items()
                if k not in task_arg_keys and not k.startswith("baseline_")
            }
            task_args = {k: v for k, v in kwargs.items() if k in task_arg_keys}
            baseline_args = {
                k.replace("baseline_", ""): v
                for k, v in kwargs.items()
                if k.startswith("baseline_")
            }

            # 2. Prepare common parameters
            dataset_names = cli_args["dataset_names"]
            model_embedding_paths = cli_args[
                "model_embedding_path"
            ]  # Always a tuple now
            output_file = cli_args["output_file"]
            compute_baseline = cli_args["compute_baseline"]
            random_seed = cli_args.get("random_seed") or 42
            datasets = get_datasets(dataset_names)

            # 3. Process task-specific arguments: load files and cast types
            processed_task_args = {}
            for name, value in task_args.items():
                param_info = invoked_task_info.task_params[name]
                # If the value is a string and looks like a path, load it from disk
                if isinstance(value, str) and "." in value:
                    try:
                        processed_task_args[name] = load_from_path(value)
                        log.info(f"Loaded task parameter '{name}' from path: {value}")
                        continue
                    except click.BadParameter:
                        pass  # It wasn't a supported file, treat as literal string
                # Otherwise, cast the value to its expected type
                processed_task_args[name] = cast_to_type(value, param_info.type)

            # 4. Add dataset-derived parameters (e.g., labels, obs)
            # This logic is simple and can live here, specific to how datasets are structured.
            if invoked_task_name == "clustering":
                # For clustering task, if obs and input_labels are not provided via files,
                # try to get them from the dataset's adata
                if "obs" not in processed_task_args:
                    processed_task_args["obs"] = datasets[0].adata.obs
                if (
                    "input_labels" not in processed_task_args
                    and hasattr(datasets[0].adata, "obs")
                    and "cell_type" in datasets[0].adata.obs.columns
                ):
                    processed_task_args["input_labels"] = datasets[0].adata.obs[
                        "cell_type"
                    ]
            # Add other task-specific dataset mappings here...

            # 5. Run the task(s)
            all_results = []

            # Run baseline if requested
            if compute_baseline:
                log.info("--- Running Baseline ---")
                # For baseline, the input is always the raw expression data
                raw_data = [d.adata.X for d in datasets]
                baseline_input = raw_data if len(raw_data) > 1 else raw_data[0]

                baseline_results = run_task(
                    task_name=invoked_task_name,
                    cell_representation=baseline_input,
                    run_baseline=True,
                    baseline_params=baseline_args,
                    task_params=processed_task_args,
                    random_seed=random_seed,
                )
                all_results.extend(baseline_results)
                log.info("--- Baseline Run Complete ---")

            # Run with model embedding
            log.info("--- Running Model ---")
            model_embeddings = [load_from_path(p) for p in model_embedding_paths]
            model_input = (
                model_embeddings if len(model_embeddings) > 1 else model_embeddings[0]
            )

            model_results = run_task(
                task_name=invoked_task_name,
                cell_representation=model_input,
                run_baseline=False,  # Never compute baseline on an embedding
                task_params=processed_task_args,
                random_seed=random_seed,
            )
            all_results.extend(model_results)
            log.info("--- Model Run Complete ---")

            # 6. Write results
            metric_objects = [MetricResult(**res) for res in all_results]
            write_results(metric_objects, output_file)
            click.echo(f"✅ Task '{invoked_task_name}' completed successfully.")

        except (click.ClickException, ValueError) as e:
            log.error(f"Task '{invoked_task_name}' failed: {e}")
            raise click.ClickException(str(e))  # Re-raise as a clean CLI error
        except Exception as e:
            log.error(
                f"An unexpected error occurred in task '{invoked_task_name}': {e}",
                exc_info=True,
            )
            raise click.ClickException(f"An unexpected error occurred: {e}")

    # --- Dynamically Add Options to the Command ---
    def add_click_options(options):
        def decorator(f):
            for option in reversed(options):
                f = option(f)
            return f

        return decorator

    # Create options from task and baseline schemas
    task_options = [
        click.option(
            f"--{name.replace('_', '-')}",
            default=info.default,
            help=f"Task parameter: {name}",
            required=info.required,
        )
        for name, info in task_info.task_params.items()
    ]
    baseline_options = [
        click.option(
            f"--baseline-{name.replace('_', '-')}",
            default=info.default,
            help=f"Baseline parameter for --compute-baseline: {name}",
        )
        for name, info in task_info.baseline_params.items()
    ]

    # Define common options for all tasks
    common_options = [
        click.option(
            "--dataset",
            "-d",
            "dataset_names",
            required=True,
            multiple=True,
            help="Name of the dataset(s) to use.",
        ),
        click.option(
            "--model-embedding-path",
            required=True,
            multiple=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Path to model embedding file(s). Provide one per dataset.",
        ),
        click.option(
            "--output-file",
            "-o",
            type=click.Path(dir_okay=False, writable=True),
            help="Path to save results (JSON format).",
        ),
        click.option(
            "--compute-baseline",
            is_flag=True,
            help="Compute and evaluate a baseline for this task.",
        ),
        click.option(
            "--random-seed", type=int, help="Random seed for reproducibility."
        ),
    ]

    # Apply all options to the callback function
    task_callback = add_click_options(common_options)(task_callback)
    task_callback = add_click_options(baseline_options)(task_callback)
    task_callback = add_click_options(task_options)(task_callback)

    run.add_command(task_callback)
