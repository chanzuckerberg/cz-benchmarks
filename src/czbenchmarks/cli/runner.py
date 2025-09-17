import logging
from typing import Any, Dict, List, Optional, Union

from anndata import AnnData

from czbenchmarks.tasks.task import TASK_REGISTRY
from czbenchmarks.tasks.types import CellRepresentation

from ..constants import RANDOM_SEED
from .resolve_reference import (
    AnnDataReference,
    is_anndata_reference,
    resolve_value_recursively,
)

logger = logging.getLogger(__name__)


def run_task(
    task_name: str,
    *,
    adata: AnnData,
    cell_representation: Union[str, CellRepresentation],
    run_baseline: bool = False,
    baseline_params: Optional[Dict[str, Any]] = None,
    task_params: Optional[Dict[str, Any]] = None,
    random_seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    logger.info(f"Preparing to run task: '{task_name}' (CLI)")
    baseline_params = baseline_params or {}
    task_params = task_params or {}

    TaskClass = TASK_REGISTRY.get_task_class(task_name)
    InputModel = TaskClass.input_model
    task_instance = TaskClass(random_seed=random_seed)

    cell_representation_for_execution = cell_representation
    if is_anndata_reference(cell_representation):
        cell_representation_for_execution = AnnDataReference.parse(
            cell_representation
        ).resolve(adata)

    params_resolved = resolve_value_recursively(task_params, adata)
    baseline_resolved = resolve_value_recursively(baseline_params, adata)

    if run_baseline:
        logger.info(f"Computing baseline for '{task_name}'...")
        try:
            cell_representation_for_execution = task_instance.compute_baseline(
                expression_data=cell_representation_for_execution, **baseline_resolved
            )
            logger.info("Baseline computation complete.")
        except NotImplementedError:
            logger.warning(
                f"Baseline calculation is not implemented for '{task_name}'."
            )
        except Exception as e:
            logger.error(f"Error during baseline computation for '{task_name}': {e}")
            raise

    TASK_REGISTRY.validate_task_input(task_name, params_resolved)
    try:
        task_input = InputModel(**params_resolved)
    except Exception as e:
        logger.error(
            f"Failed to create TaskInput for '{task_name}' with params {params_resolved}. Error: {e}"
        )
        raise ValueError(f"Invalid task parameters for '{task_name}': {e}") from e

    logger.info(f"Executing task logic for '{task_name}' (CLI)...")
    results = task_instance.run(
        cell_representation=cell_representation_for_execution, task_input=task_input
    )
    logger.info(f"Task '{task_name}' execution complete.")

    return [res.model_dump() for res in results]
