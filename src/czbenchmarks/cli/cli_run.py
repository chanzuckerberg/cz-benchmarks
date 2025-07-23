import click
import logging

from .registry import TASK_REGISTRY, get_task_def
from .utils import (
    add_options,
    get_datasets,
    load_embedding,
    prepare_task_inputs,
    write_results
)
from ..metrics.types import MetricResult

log = logging.getLogger(__name__)

# Common options to be applied to all task sub-commands
def common_task_options(func):
    """A decorator to apply common options to a task command."""
    func = click.option(
        "--dataset", "-d", "dataset_names",
        required=True,
        multiple=True,
        help="Name of the dataset(s) to use. Can be specified multiple times."
    )(func)
    func = click.option(
        "--model-embedding-path",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Path to the model embedding file (e.g., .npy or .csv)."
    )(func)
    func = click.option(
        "--output-file", "-o",
        type=click.Path(dir_okay=False, writable=True),
        help="Path to file to save results (JSON format)."
    )(func)
    func = click.option(
        "--compute-baseline",
        is_flag=True,
        help="Compute and evaluate a baseline for this task."
    )(func)
    func = click.option(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility."
    )(func)
    return func


# Central 'run' command group
@click.group(invoke_without_command=True)
@click.pass_context
def run(ctx):
    """Run a benchmark task."""
    if ctx.invoked_subcommand is None:
        click.echo("Please specify a task to run, e.g., 'czbenchmarks run clustering'.")
        click.echo(f"Available tasks: {', '.join(TASK_REGISTRY.keys())}")


# Dynamically create and add a sub-command for each task in the registry
for task_name, task_def in TASK_REGISTRY.items():

    @click.command(name=task_name, help=task_def.description)
    @add_options(task_def.parameters)
    @add_options(task_def.baseline_parameters, is_baseline=True)
    @common_task_options
    def task_callback(**kwargs):
        """Generic callback for all dynamically generated task commands."""
        ctx = click.get_current_context()
        invoked_task_name = ctx.command.name
        task_def = get_task_def(invoked_task_name)
        log.info(f"Starting task: '{invoked_task_name}'")

        # Separate CLI args
        dataset_names = kwargs.pop('dataset_names')
        model_embedding_path = kwargs.pop('model_embedding_path')
        output_file = kwargs.pop('output_file')
        compute_baseline = kwargs.pop('compute_baseline')
        random_seed = kwargs.pop('random_seed', None)
        
        baseline_args = {}
        for p in task_def.baseline_parameters:
            kwarg_name = f"baseline_{p.name}"
            if kwarg_name in kwargs:
                baseline_args[p.name] = kwargs.pop(kwarg_name)

        # 1. Load datasets and model embedding
        datasets = get_datasets(dataset_names)
        model_embedding = load_embedding(model_embedding_path)
        log.info(f"Loaded {len(datasets)} dataset(s) and model embedding from '{model_embedding_path}'.")

        # 2. Prepare and validate inputs for the task
        task_input = prepare_task_inputs(task_def, datasets, kwargs)
        
        # 3. Instantiate task
        task_instance = task_def.task_class(random_seed=random_seed)

        # 4. Run task on the primary model embedding
        log.info("Running task on provided model embedding...")
        all_metric_results: list[MetricResult] = task_instance.run(
            cell_representation=model_embedding,
            task_input=task_input,
        )
        log.info(f"Computed {len(all_metric_results)} metric(s) for the model embedding.")

        # 5. Compute and run baseline if requested
        if compute_baseline:
            log.info("Computing and running baseline...")
            # Baseline is computed on raw expression data from the first dataset
            # For multi-dataset tasks, this logic would need to be adapted.
            raw_expression = datasets[0].adata.X
            
            baseline_embedding = task_instance.compute_baseline(
                expression_data=raw_expression, **baseline_args
            )
            log.info("Running task on computed baseline embedding...")
            baseline_metric_results = task_instance.run(
                cell_representation=baseline_embedding,
                task_input=task_input,
            )
            all_metric_results.extend(baseline_metric_results)
            log.info(f"Computed {len(baseline_metric_results)} metric(s) for the baseline.")

        # 6. Write results
        write_results(all_metric_results, output_file)
        click.echo(f"✅ Task '{invoked_task_name}' completed successfully.")

    # Attach the dynamically created command to the 'run' group
    run.add_command(task_callback)