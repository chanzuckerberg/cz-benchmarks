import click
import logging
from .cli_list import list_cmd
from .cli_run import run
from .utils import get_version


@click.group()
@click.version_option(version=get_version(), prog_name="czbenchmarks")
@click.option(
    "--log-level",
    "-ll",
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"], case_sensitive=False
    ),
    default="info",
    help="Set the logging level.",
)
def main(log_level: str):
    """
    czbenchmarks: A command-line utility for using cz-benchmarks.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


# Add subcommands to the main group
main.add_command(list_cmd)
main.add_command(run)

if __name__ == "__main__":
    main()
