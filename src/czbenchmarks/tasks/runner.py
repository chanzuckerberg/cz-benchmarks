import logging
from typing import Any, Dict, List, Union

from ..constants import RANDOM_SEED
from .task import TASK_REGISTRY
from .types import CellRepresentation

log = logging.getLogger(__name__)


def run_task(
    task_name: str,
    cell_representation: Union[CellRepresentation, List[CellRepresentation]],
    run_baseline: bool = False,
    baseline_params: Dict[str, Any] = None,
    task_params: Dict[str, Any] = None,
    random_seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Runs a benchmark task with the given parameters. This is the primary API method for Task execution.

    This function expects all inputs to be fully-formed Python objects (e.g., numpy arrays,
    pandas DataFrames), not file paths. The calling layer (e.g., a CLI) is responsible
    for loading data from disk.

    Args:
        task_name: The normalized name of the task to run (e.g., 'clustering').
        cell_representation: The primary data for the task. For a standard run, this is a
            model embedding. For a baseline run, this is the raw expression data.
            For multi-dataset tasks, this is a list of embeddings or raw data.
        run_baseline: If True, `cell_representation` is treated as raw data to compute a
            baseline embedding before running the task.
        baseline_params: A dictionary of parameters for the task's `compute_baseline` method.
            Only used if `run_baseline` is True.
        task_params: A dictionary of parameters for the task's specific `TaskInput` model.
        random_seed: An integer for reproducibility.

    Returns:
        A list of metric results, where each result is a dictionary.
    """
    log.info(f"Preparing to run task: '{task_name}'")
    task_params = task_params or {}
    baseline_params = baseline_params or {}

    # Get the task class and its Pydantic input model from the registry
    TaskClass = TASK_REGISTRY.get_task_class(task_name)
    InputModel = TaskClass.input_model
    task_instance = TaskClass(random_seed=random_seed)

    # Prepare the embedding for the task run.
    # If `run_baseline` is True, we first compute the baseline embedding.
    # The result is then used as the input for the main task logic.
    rep_for_run = cell_representation
    if run_baseline:
        log.info(f"Computing baseline for '{task_name}'...")
        try:
            rep_for_run = task_instance.compute_baseline(
                expression_data=cell_representation, **baseline_params
            )
            log.info("Baseline computation complete.")
        except NotImplementedError:
            log.warning(f"Baseline calculation is not implemented for '{task_name}'.")
            # If baseline is not implemented, we proceed using the original representation.
        except Exception as e:
            log.error(f"Error during baseline computation for '{task_name}': {e}")
            raise

    # Instantiate the task-specific Pydantic input model.
    try:
        task_input = InputModel(**task_params)
    except Exception as e:
        log.error(
            f"Failed to create TaskInput for '{task_name}' with params {task_params}. Error: {e}"
        )
        raise ValueError(f"Invalid task parameters for '{task_name}': {e}") from e

    # Run the task's main logic
    log.info(f"Executing task logic for '{task_name}'...")
    results = task_instance.run(cell_representation=rep_for_run, task_input=task_input)
    log.info(f"Task '{task_name}' execution complete.")

    # Return results as dictionaries for easy serialization
    return [res.model_dump() for res in results]
