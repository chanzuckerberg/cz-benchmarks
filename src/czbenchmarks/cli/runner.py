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
    """
    Orchestrates the execution of a benchmark task, including baseline computation.
    """
    logger.info(f"Preparing to run task: '{task_name}'")
    baseline_params = baseline_params or {}
    task_params = task_params or {}

    # Get task-specific classes from the registry
    TaskClass = TASK_REGISTRY.get_task_class(task_name)
    InputModel = TaskClass.input_model
    BaselineModel = TaskClass.baseline_model
    task_instance = TaskClass(random_seed=random_seed)

    # Resolve the primary cell representation (either from AnnData or an external file)
    model_representation = cell_representation
    if is_anndata_reference(cell_representation):
        model_representation = AnnDataReference.parse(str(cell_representation)).resolve(
            adata
        )

    # Resolve any AnnData references within the task parameters
    task_params_resolved = resolve_value_recursively(task_params, adata)

    # Validate and create the TaskInput object
    TASK_REGISTRY.validate_task_input(task_name, task_params_resolved)
    try:
        task_input = InputModel(**task_params_resolved)
    except Exception as e:
        logger.error(f"Failed to create TaskInput for '{task_name}'. Error: {e}")
        raise ValueError(f"Invalid task parameters for '{task_name}': {e}") from e

    results = []

    # --- Run on Model Representation ---
    logger.info(f"Executing task '{task_name}' on the provided model representation...")
    model_results = task_instance.run(
        cell_representation=model_representation, task_input=task_input
    )
    # Add metadata to distinguish model results
    for res in model_results:
        res.params["run_type"] = "model"
    results.extend(model_results)
    logger.info("Model evaluation complete.")

    # --- Optionally Run on Baseline Representation ---
    if run_baseline:
        logger.info(f"Computing baseline for '{task_name}'...")
        try:
            # Resolve AnnData references in baseline parameters
            baseline_params_resolved = resolve_value_recursively(baseline_params, adata)
            # Create the BaselineInput object
            baseline_input = BaselineModel(**baseline_params_resolved)

            # The baseline is always computed from the raw expression data in adata.X
            baseline_representation = task_instance.compute_baseline(
                expression_data=adata.X, baseline_input=baseline_input
            )
            logger.info("Baseline computation complete.")

            # Run the same task evaluation on the baseline representation
            logger.info(
                f"Executing task '{task_name}' on the baseline representation..."
            )
            baseline_results = task_instance.run(
                cell_representation=baseline_representation, task_input=task_input
            )
            # Add metadata to distinguish baseline results
            for res in baseline_results:
                res.params["run_type"] = "baseline"
            results.extend(baseline_results)
            logger.info("Baseline evaluation complete.")

        except NotImplementedError:
            logger.warning(
                f"Baseline computation is not implemented for '{task_name}'. Skipping baseline run."
            )
        except Exception as e:
            logger.error(f"Error during baseline computation for '{task_name}': {e}")
            raise

    logger.info(f"All executions for task '{task_name}' are complete.")
    return [res.model_dump() for res in results]
