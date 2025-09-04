import json

import click
from rich.console import Console
from rich.table import Table

from czbenchmarks.tasks.task import TASK_REGISTRY

from ..datasets import utils as dataset_utils


@click.command(name="list")
@click.argument("list_type", type=click.Choice(["datasets", "tasks"]))
@click.option(
    "-f",
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
        tasks = []
        for name in TASK_REGISTRY.list_tasks():
            try:
                task_info = TASK_REGISTRY.get_task_info(name)
                tasks.append({"name": task_info.display_name, "description": task_info.description})
            except Exception as e:
                # Optionally log or print the error, but skip the broken task
                tasks.append({"name": name, "description": f"Error: {e}"})
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
