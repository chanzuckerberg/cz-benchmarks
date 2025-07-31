import click
from rich.console import Console
from rich.table import Table
import json
from .registry import TASK_REGISTRY
from ..datasets import utils as dataset_utils


@click.command(name="list")
@click.argument("list_type", type=click.Choice(["datasets", "tasks"]))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"], case_sensitive=False),
    default="table",
    help="Output format: json or table (default: table)",
)
def list_cmd(list_type: str, output_format: str):
    """List available datasets or tasks."""
    console = Console()

    if list_type == "tasks":
        tasks = [
            {"name": name, "description": task_def.description}
            for name, task_def in TASK_REGISTRY.items()
        ]
        if output_format == "json":
            console.print(json.dumps(tasks, indent=2))
        else:
            table = Table(title="Available Tasks")
            table.add_column("Name", no_wrap=True)
            table.add_column("Description")
            for task in tasks:
                table.add_row(task["name"], task["description"])
            console.print(table)

    elif list_type == "datasets":
        datasets = dataset_utils.list_available_datasets()
        if output_format == "json":
            print(datasets)  # Print directly for JSON output
            console.print(json.dumps(datasets, indent=2))
        else:
            table = Table(title="Available Datasets", show_lines=True)
            table.add_column("Dataset", no_wrap=True)
            table.add_column("Organism", no_wrap=True)
            table.add_column("URL", overflow="fold")

            # Add rows for each dataset
            for dataset in datasets:
                table.add_row(
                    dataset, datasets[dataset]["organism"], datasets[dataset]["url"]
                )
            console.print(table)
