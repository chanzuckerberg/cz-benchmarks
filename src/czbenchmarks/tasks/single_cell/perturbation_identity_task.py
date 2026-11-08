"""Native CZI task for preaggregated held-out perturbation responses.

This is distinct from cell-level perturbation expression prediction. It scores
explicitly supplied outcome and prediction matrices; it neither fits nor calls a
model. All provenance and holdout declarations are caller-supplied, not audited.
"""

import hashlib
import json
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...tasks.types import CellRepresentation
from ..task import NoBaselineInput, Task, TaskInput, TaskOutput
from .perturbation_identity_response import (
    IdentityPredictions,
    LabeledResponses,
    _check_response,
    compare_identity_predictions,
)


class PerturbationIdentityTaskInput(TaskInput):
    """Inputs for a single prespecified context of donor-target responses.

    All response matrices have independently labeled donor-target rows and gene
    columns. Matched prediction values are the ``cell_representation`` passed to
    ``Task.run``; its labels, scale and identity inputs are given below. These
    rows are pseudobulk response units, never synthetic single cells.

    ``fitted_donors`` maps each of ``matched``, ``alternative``, ``generic`` and
    ``learned_preprocessing`` to a mapping of each evaluation donor to the
    donor IDs used when fitting that component for that donor's fold. The last
    list can be empty if no preprocessing was learned. The evaluator verifies
    the declaration excludes the corresponding evaluation donor, but cannot
    verify that a model, hyperparameter or feature selection was actually fit
    as declared. The caller must report permitted held-out controls separately.
    """

    truth: LabeledResponses
    matched_units: pd.MultiIndex
    matched_genes: pd.Index
    matched_scale: str
    matched_identity_inputs: List[str]
    alternative: IdentityPredictions
    generic: LabeledResponses
    fitted_donors: Dict[str, Dict[str, List[str]]]
    dataset_id: str = Field(description="Evaluation dataset identifier.")
    context: str = Field(description="Prespecified biological context for all rows.")
    matched_model_id: str = Field(description="Identifier for the matched predictor.")
    alternative_reference_id: str = Field(
        description="Identifier for the alternative prediction set."
    )
    generic_reference_id: str = Field(
        description="Identifier for the generic predictor."
    )
    heldout_control_usage: str = Field(
        description="Declared access to held-out controls."
    )
    alternative_kind: Literal["model_input_rerun", "output_reassignment"]
    alternative_provenance: str = Field(
        description="How alternative predictions were generated."
    )
    generic_provenance: str = Field(
        description="How the generic reference was trained and generated."
    )


class PerturbationIdentityTaskOutput(TaskOutput):
    """Validated donor-balanced response losses and paired differences."""

    overall: Dict[str, float]
    per_donor: Dict[str, Dict[str, float]]
    panel_digest: str


def _align_response(
    truth: LabeledResponses, x: LabeledResponses, name: str
) -> np.ndarray:
    """Align exactly the same labeled rows and genes; never silently intersect."""
    _check_response(name, x)
    if x.scale != truth.scale:
        raise ValueError(f"{name} response scale must match truth")
    rows = x.units.get_indexer(truth.units)
    genes = x.genes.get_indexer(truth.genes)
    if len(x.units) != len(truth.units) or (rows < 0).any():
        raise ValueError(
            f"{name} must have exactly the same evaluated donor-target units as truth"
        )
    if len(x.genes) != len(truth.genes) or (genes < 0).any():
        raise ValueError(f"{name} must have exactly the same gene panel as truth")
    return np.asarray(x.values)[np.ix_(rows, genes)]


def _check_fitted_donors(
    task_input: PerturbationIdentityTaskInput, donors: np.ndarray
) -> None:
    expected = {"matched", "alternative", "generic", "learned_preprocessing"}
    if set(task_input.fitted_donors) != expected:
        raise ValueError(
            f"fitted_donors must declare exactly these components: {sorted(expected)}"
        )
    evaluation_donors = set(donors)
    for component, folds in task_input.fitted_donors.items():
        if set(folds) != evaluation_donors:
            raise ValueError(
                f"{component} must declare fitting donors for every evaluation donor"
            )
        for heldout, training in folds.items():
            if any(not isinstance(d, str) or not d.strip() for d in training):
                raise ValueError("fitted donor IDs must be nonempty strings")
            if len(training) != len(set(training)):
                raise ValueError("fitted donor IDs must not contain duplicates")
            if heldout in training:
                raise ValueError(
                    f"held-out donor {heldout!r} appears in {component} fitting donors"
                )
            if component != "learned_preprocessing" and not training:
                raise ValueError(
                    f"{component} requires at least one declared training donor"
                )
    matched = task_input.fitted_donors["matched"]
    alternative = task_input.fitted_donors["alternative"]
    if any(set(matched[d]) != set(alternative[d]) for d in evaluation_donors):
        raise ValueError(
            "matched and alternative predictions must declare identical fitted donors per fold"
        )
    for name in (
        "dataset_id",
        "context",
        "matched_model_id",
        "alternative_reference_id",
        "generic_reference_id",
        "heldout_control_usage",
        "alternative_provenance",
        "generic_provenance",
    ):
        if not getattr(task_input, name).strip():
            raise ValueError(f"{name} must be a nonempty declaration")


class PerturbationIdentityTask(Task):
    """Score preaggregated donor-held-out response predictions in one context.

    The matched, alternative and generic responses must be supplied independently.
    A generic output must be identical across targets within each donor. This is
    an output check, not proof that the predictor did not use identity or held-out
    outcomes. A declared rerun is not verified by the task. Report alternative
    input reruns and output reassignment controls as distinct comparisons.
    """

    display_name = "Perturbation Identity Response"
    description = "Compare matched, alternative-identity, and generic response predictions across held-out donors."
    input_model = PerturbationIdentityTaskInput
    baseline_model = NoBaselineInput

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: PerturbationIdentityTaskInput,
    ) -> PerturbationIdentityTaskOutput:
        if not isinstance(cell_representation, np.ndarray):
            raise ValueError(
                "This task requires a dense numpy response matrix, not cells or embeddings"
            )
        truth = task_input.truth
        _check_response("truth", truth)
        matched_responses = LabeledResponses(
            cell_representation,
            task_input.matched_units,
            task_input.matched_genes,
            task_input.matched_scale,
        )
        matched = IdentityPredictions(
            matched_responses, task_input.matched_identity_inputs
        )
        # This also validates identity declarations, masks, numeric values and overflow.
        alt_result = compare_identity_predictions(
            truth, matched, task_input.alternative
        )
        matched_values = _align_response(truth, matched_responses, "matched")
        generic_values = _align_response(truth, task_input.generic, "generic")
        groups = np.asarray(truth.units.get_level_values(0), dtype=object)
        _check_fitted_donors(task_input, groups)
        for donor in sorted(set(groups)):
            rows = generic_values[groups == donor]
            if not np.array_equal(rows, np.broadcast_to(rows[0], rows.shape)):
                raise ValueError(
                    f"generic predictions must be identical across targets within donor {donor!r}"
                )
        true_values = np.asarray(truth.values)
        # Registry validates all three matrices are finite and checks loss overflow.
        generic_gain = metrics_registry.compute(
            MetricType.PAIRED_MSE_REDUCTION,
            y_true=true_values,
            y_pred=matched_values,
            y_reference=generic_values,
            group_ids=groups.tolist(),
        )
        alt_values = _align_response(
            truth, task_input.alternative.responses, "alternative"
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                y = np.asarray(true_values, dtype=np.float64)
                arrays = {
                    "matched_mse": matched_values,
                    "alternative_mse": alt_values,
                    "generic_mse": generic_values,
                }
                row_losses = {
                    name: np.mean(
                        np.square(y - np.asarray(value, dtype=np.float64)), axis=1
                    )
                    for name, value in arrays.items()
                }
                per_donor = {}
                for donor in sorted(set(groups)):
                    losses = {
                        name: float(np.mean(rows[groups == donor]))
                        for name, rows in row_losses.items()
                    }
                    losses["alternative_minus_matched_mse"] = alt_result.by_donor[donor]
                    losses["generic_minus_matched_mse"] = (
                        losses["generic_mse"] - losses["matched_mse"]
                    )
                    per_donor[donor] = losses
                generic_mse = float(
                    np.mean([v["generic_mse"] for v in per_donor.values()])
                )
        except FloatingPointError as exc:
            raise ValueError(
                "Absolute MSE overflowed; check the response scale"
            ) from exc
        if not np.isfinite(generic_mse) or any(
            not np.isfinite(value)
            for values in per_donor.values()
            for value in values.values()
        ):
            raise ValueError("Absolute MSE produced a nonfinite value")
        return PerturbationIdentityTaskOutput(
            overall={
                "matched_mse": alt_result.matched_mse,
                "alternative_mse": alt_result.alternative_mse,
                "generic_mse": generic_mse,
                "alternative_minus_matched_mse": alt_result.paired_mse_reduction,
                "generic_minus_matched_mse": generic_gain,
            },
            per_donor=per_donor,
            panel_digest=hashlib.sha256(
                json.dumps(
                    {
                        "units": sorted(list(truth.units)),
                        "genes": sorted(list(truth.genes)),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def _compute_metrics(
        self,
        task_input: PerturbationIdentityTaskInput,
        task_output: PerturbationIdentityTaskOutput,
    ) -> List[MetricResult]:
        results = []
        for donor, values in [
            (None, task_output.overall),
            *task_output.per_donor.items(),
        ]:
            for name, value in values.items():
                metric_type = (
                    MetricType.PAIRED_MSE_REDUCTION
                    if name.endswith("minus_matched_mse")
                    else MetricType.MEAN_SQUARED_ERROR
                )
                params = {
                    "measure": name,
                    "aggregation": "equal_donor" if donor is None else "within_donor",
                    "dataset_id": task_input.dataset_id,
                    "context": task_input.context,
                    "response_scale": task_input.truth.scale,
                    "panel_sha256": task_output.panel_digest,
                    "matched_model_id": task_input.matched_model_id,
                }
                if donor is not None:
                    params["donor"] = donor
                if name.startswith("alternative"):
                    params["alternative_kind"] = task_input.alternative_kind
                    params["reference_id"] = task_input.alternative_reference_id
                if name.startswith("generic"):
                    params["reference_id"] = task_input.generic_reference_id
                results.append(
                    MetricResult(metric_type=metric_type, value=value, params=params)
                )
        return results

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        baseline_input: NoBaselineInput = None,
    ):
        raise NotImplementedError(
            "Supply a separately fitted generic reference in the task input."
        )
