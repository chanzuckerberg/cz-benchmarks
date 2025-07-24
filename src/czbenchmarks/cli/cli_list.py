import click
from .registry import TASK_REGISTRY
from ..datasets import utils as dataset_utils


@click.command(name="list")
@click.argument("list_type", type=click.Choice(["datasets", "tasks"]))
def list_cmd(list_type: str):
    """List available datasets or tasks."""
    if list_type == "tasks":
        for name, task_def in TASK_REGISTRY.items():
            click.echo(f"- {name}: {task_def.description}")
    elif list_type == "datasets":
        datasets = dataset_utils.list_available_datasets()
        click.echo("\n".join(datasets))
