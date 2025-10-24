"""
Unified task execution module for benchmarking tasks.

This module provides a single, robust task runner that supports both single and
multi-dataset benchmarks, with automatic reference resolution and baseline computation.
"""

import inspect
import logging
from typing import Any, Dict, List, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from czbenchmarks.tasks import TASK_REGISTRY
from czbenchmarks.tasks.types import CellRepresentation

from .resolve_reference import resolve_anndata_references

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def _ensure_dense_matrix(cell_rep: Union[CellRepresentation, np.ndarray]) -> np.ndarray:
    if sp.issparse(cell_rep):
        return cell_rep.toarray()
    return cell_rep


def run_task(
    task_name: str,
    *,
    adata_input: Union[AnnData, List[AnnData]],
    cell_representation_input: Union[
        str, CellRepresentation, List[Union[str, CellRepresentation]]
    ],
    run_baseline: bool = False,
    baseline_params: Dict[str, Any] | None = None,
    task_params: Dict[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Unified task runner for single and multi-dataset benchmarks.
    """

    if random_seed is None:
        random_seed = RANDOM_SEED

    logger.debug(f"Preparing to run task: '{task_name}'")

    TaskClass = TASK_REGISTRY.get_task_class(task_name)
    task_instance = TaskClass(random_seed=random_seed)
    is_multi_dataset_task = getattr(task_instance, "requires_multiple_datasets", False)

    if isinstance(adata_input, list):
        anndata_objects = [
            dataset.adata if hasattr(dataset, "adata") else dataset
            for dataset in adata_input
        ]
    else:
        anndata_objects = (
            adata_input.adata if hasattr(adata_input, "adata") else adata_input
        )

    resolved_task_params = resolve_anndata_references(
        task_params or {}, anndata_objects
    )
    resolved_baseline_params = resolve_anndata_references(
        baseline_params or {}, anndata_objects
    )
    resolved_cell_repr = resolve_anndata_references(
        cell_representation_input, anndata_objects
    )

    if run_baseline:
        if is_multi_dataset_task:
            logger.warning(
                f"Baseline computation not supported for multi-dataset task '{task_name}'."
            )
        else:
            logger.info(f"Computing baseline for '{task_name}'...")

            baseline_signature = inspect.signature(task_instance.compute_baseline)
            params = baseline_signature.parameters

            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            _exclude = {"self", "expression_data", "cell_representation"}

            if accepts_var_kwargs:
                filtered_baseline_params = dict(resolved_baseline_params)
            else:
                allowed = {
                    name
                    for name, p in params.items()
                    if name not in _exclude
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                filtered_baseline_params = {
                    k: v
                    for k, v in (resolved_baseline_params or {}).items()
                    if k in allowed
                }

            expr_for_baseline = anndata_objects.X
            try:
                logger.debug(
                    f"Baseline arguments: expression_data shape={expr_for_baseline.shape}, "
                    f"params={', '.join(f'{k}={v!r}' for k, v in (filtered_baseline_params or {}).items())}"
                )
                resolved_cell_repr = task_instance.compute_baseline(
                    expression_data=expr_for_baseline, **filtered_baseline_params
                )
                logger.info("Baseline computation complete.")
            except NotImplementedError:
                logger.warning(f"Baseline not implemented for '{task_name}'.")
            except Exception as e:
                logger.warning(
                    f"Baseline computation failed for '{task_name}': {e}. "
                    f"Continuing with original cell representation."
                )

    final_cell_repr = (
        [_ensure_dense_matrix(rep) for rep in resolved_cell_repr]
        if isinstance(resolved_cell_repr, list)
        else _ensure_dense_matrix(resolved_cell_repr)
    )

    try:
        task_input_model = TaskClass.input_model(**resolved_task_params)
    except Exception as e:
        logger.error(f"Failed to create task input model for '{task_name}': {e}")
        raise

    logger.debug(
        f"Executing task {task_name} with arguments: cell_representation shape={final_cell_repr.shape if hasattr(final_cell_repr, 'shape') else type(final_cell_repr)}, "
        f"task_input={task_input_model}"
    )
    results = task_instance.run(
        cell_representation=final_cell_repr,
        task_input=task_input_model,
    )
    logger.info(f"Task '{task_name}' execution complete.")
    return [res.model_dump() for res in results]
