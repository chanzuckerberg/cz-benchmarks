import click
import logging
import numpy as np

from .registry import TASK_REGISTRY, get_task_def
from .utils import (
    add_options,
    discover_task_parameters,
    discover_baseline_parameters,
    get_datasets,
    load_embedding,
    prepare_task_inputs,
    write_results,
)
from ..metrics.types import MetricResult

log = logging.getLogger(__name__)


def common_task_options(func):
    """A decorator to apply common options to a task command."""
    func = click.option(
        "--dataset",
        "-d",
        "dataset_names",
        required=True,
        multiple=True,
        help="Name of the dataset(s) to use.",
    )
    func = click.option(
        "--model-embedding-path",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to the model embedding file (.npy or .csv).",
    )
    func = click.option(
        "--output-file",
        "-o",
        type=click.Path(dir_okay=False, writable=True),
        help="Path to save results (JSON format).",
    )
    func = click.option(
        "--compute-baseline",
        is_flag=True,
        help="Compute and evaluate a baseline for this task.",
    )
    func = click.option(
        "--random-seed", type=int, help="Random seed for reproducibility."
    )
    return func


@click.group(invoke_without_command=True)
@click.pass_context
def run(ctx):
    """Run a benchmark task."""
    if ctx.invoked_subcommand is None:
        click.echo("Please specify a task to run, e.g., 'czbenchmarks run clustering'.")
        click.echo(f"Available tasks: {', '.join(TASK_REGISTRY.keys())}")


for task_name, task_def in TASK_REGISTRY.items():
    # Discover parameters at definition time
    task_params = discover_task_parameters(task_def)
    baseline_params = discover_baseline_parameters(task_def)

    @click.command(name=task_name, help=task_def.description)
    @add_options(task_params)
    @add_options(baseline_params)
    @common_task_options
    def task_callback(**kwargs):
        ctx = click.get_current_context()
        invoked_task_name = ctx.command.name
        task_def = get_task_def(invoked_task_name)
        log.info(f"Starting task: '{invoked_task_name}'")

        dataset_names = kwargs.pop("dataset_names")
        model_embedding_path = kwargs.pop("model_embedding_path")
        output_file = kwargs.pop("output_file")
        compute_baseline = kwargs.pop("compute_baseline")
        random_seed = kwargs.pop("random_seed", None)

        baseline_args = {
            k.replace("baseline_", ""): v
            for k, v in kwargs.items()
            if k.startswith("baseline_")
        }
        task_args = {k: v for k, v in kwargs.items() if not k.startswith("baseline_")}

        datasets = get_datasets(dataset_names)
        model_embedding = load_embedding(model_embedding_path)
        log.info(f"Loaded {len(datasets)} dataset(s) and model embedding.")

        task_input = prepare_task_inputs(task_def, datasets, task_args)

        task_instance = task_def.task_class(random_seed=random_seed)

        log.info("Running task on provided model embedding...")
        all_metric_results: list[MetricResult] = task_instance.run(
            cell_representation=model_embedding,
            task_input=task_input,
        )
        log.info(
            f"Computed {len(all_metric_results)} metric(s) for the model embedding."
        )

        if compute_baseline:
            log.info("Computing and running baseline...")

            if task_def.requires_multiple_datasets:
                raw_expression = np.vstack([d.adata.X for d in datasets])
            else:
                raw_expression = datasets[0].adata.X

            # Special context-dependent args for perturbation baseline
            if invoked_task_name == "perturbation":
                baseline_args["var_names"] = datasets[0].adata.var_names
                baseline_args["obs_names"] = datasets[0].adata.obs_names

            baseline_embedding = task_instance.compute_baseline(
                cell_representation=raw_expression, **baseline_args
            )
            log.info("Running task on computed baseline embedding...")
            baseline_metric_results = task_instance.run(
                cell_representation=baseline_embedding,
                task_input=task_input,
            )
            all_metric_results.extend(baseline_metric_results)
            log.info(
                f"Computed {len(baseline_metric_results)} metric(s) for the baseline."
            )

        write_results(all_metric_results, output_file)
        click.echo(f"✅ Task '{invoked_task_name}' completed successfully.")

    run.add_command(task_callback)
