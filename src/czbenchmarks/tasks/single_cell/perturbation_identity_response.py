"""Compare supplied response predictions under matched and alternative identities.

This adapter is deliberately separate from the cell-level masked-expression task.
It does not run a model, construct alternative inputs, choose controls, or fit a
reference. For an input-sensitivity claim, the caller must supply genuinely
generated alternative-input predictions and document that all other inputs were
held fixed. Labels and scale declarations are checked, but provenance, leakage,
and biological validity cannot be verified.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...metrics import metrics_registry
from ...metrics.types import MetricType


@dataclass(frozen=True)
class LabeledResponses:
    """Pre-aggregated responses, with one row per (donor, target) and one column per gene.

    ``units`` is a two-level index in (donor, evaluated target) order. All three
    matrices' rows refer to the *same evaluated outcomes*, including the alternative
    prediction rows: these are not alternative-target ground truths. ``scale`` must
    describe the response transformation and control subtraction, and must match.
    Supply one prespecified context or stratum per call: this two-level index
    cannot distinguish repeated donor-target units across contexts.
    """

    values: np.ndarray
    units: pd.MultiIndex
    genes: pd.Index
    scale: str


@dataclass(frozen=True)
class IdentityPredictions:
    """Responses and the target identity actually supplied for each evaluated row.

    Identity inputs are in this matrix's row order. The declaration cannot itself
    establish that the supplied array came from an actual model-input rerun.
    """

    responses: LabeledResponses
    identity_inputs: Sequence[str]


@dataclass(frozen=True)
class IdentitySensitivityResult:
    """Matched-minus-alternative predictive advantage, weighted equally by donor.

    The value is alternative MSE minus matched MSE. Positive values mean the
    matched predictions are more accurate for this declared comparison; this is
    neither a causal effect nor statistical evidence of population generalization.
    """

    paired_mse_reduction: float
    by_donor: dict[str, float]
    matched_mse: float
    alternative_mse: float
    n_units: int
    n_genes: int


def _check_response(name: str, x: LabeledResponses) -> None:
    if not isinstance(x.units, pd.MultiIndex) or x.units.nlevels != 2:
        raise ValueError(f"{name}.units must be a two-level (donor, target) MultiIndex")
    if not x.units.is_unique or len(x.units) == 0:
        raise ValueError(f"{name}.units must be nonempty and unique")
    if any(not isinstance(v, str) or not v.strip() for row in x.units for v in row):
        raise ValueError(
            f"{name}.units must contain nonempty string donor and target IDs"
        )
    if not isinstance(x.genes, pd.Index) or isinstance(x.genes, pd.MultiIndex):
        raise TypeError(f"{name}.genes must be a one-level pandas Index")
    if not x.genes.is_unique or len(x.genes) == 0:
        raise ValueError(f"{name}.genes must be nonempty and unique")
    if any(not isinstance(v, str) or not v.strip() for v in x.genes):
        raise ValueError(f"{name}.genes must contain nonempty string IDs")
    if not isinstance(x.scale, str) or not x.scale.strip():
        raise ValueError(f"{name}.scale must be a nonempty string")
    if np.ma.isMaskedArray(x.values) and np.ma.getmaskarray(x.values).any():
        raise ValueError(f"{name}.values must not contain masked values")
    # Numeric and finite checks are also enforced by the registered metric.
    if np.shape(x.values) != (len(x.units), len(x.genes)):
        raise ValueError(f"{name}.values shape must match its labeled rows and genes")


def compare_identity_predictions(
    truth: LabeledResponses,
    matched: IdentityPredictions,
    alternative: IdentityPredictions,
) -> IdentitySensitivityResult:
    """Compare actual supplied predictions on an identical labeled response panel.

    No genes or units are silently dropped: each matrix must have exactly the same
    unique donor-target pairs and genes, although ordering may differ. The matched
    identity must equal the evaluated target and the alternative must differ in
    every row. Predictions are reindexed to the *truth* row and gene order. The
    registered paired MSE metric then averages rows within donors and donors equally.

    The caller is responsible for the disjoint train/test split, matched context,
    input rerun, gene selection, response scale and controls, as well as uncertainty.
    This function verifies declarations, not the model's execution history. It
    never derives alternate predictions by shuffling predictions or observations.
    """
    _check_response("truth", truth)
    aligned = []
    for name, pred in (("matched", matched), ("alternative", alternative)):
        x = pred.responses
        _check_response(name, x)
        if x.scale != truth.scale:
            raise ValueError(f"{name} response scale must match truth")
        row_order = x.units.get_indexer(truth.units)
        gene_order = x.genes.get_indexer(truth.genes)
        if len(x.units) != len(truth.units) or (row_order < 0).any():
            raise ValueError(
                f"{name} must have exactly the same evaluated donor-target units as truth"
            )
        if len(x.genes) != len(truth.genes) or (gene_order < 0).any():
            raise ValueError(f"{name} must have exactly the same gene panel as truth")
        if (
            np.ma.isMaskedArray(pred.identity_inputs)
            and np.ma.getmaskarray(pred.identity_inputs).any()
        ):
            raise ValueError(f"{name} identity inputs must not contain masked values")
        identities = np.asarray(pred.identity_inputs, dtype=object)
        if identities.ndim != 1 or len(identities) != len(x.units):
            raise ValueError(f"{name} must declare one identity input per row")
        if any(not isinstance(v, str) or not v.strip() for v in identities):
            raise ValueError(f"{name} identity inputs must be nonempty strings")
        identities = identities[row_order]
        target = np.asarray(truth.units.get_level_values(1), dtype=object)
        if name == "matched" and not np.all(identities == target):
            raise ValueError("matched identity inputs must match evaluated targets")
        if name == "alternative" and not np.all(identities != target):
            raise ValueError(
                "alternative identity inputs must differ from evaluated targets"
            )
        aligned.append(np.asarray(x.values)[np.ix_(row_order, gene_order)])

    groups = np.asarray(truth.units.get_level_values(0), dtype=object)

    def score(rows):
        return metrics_registry.compute(
            MetricType.PAIRED_MSE_REDUCTION,
            y_true=np.asarray(truth.values)[rows],
            y_pred=aligned[0][rows],
            y_reference=aligned[1][rows],
            group_ids=groups[rows].tolist(),
        )

    all_rows = np.arange(len(truth.units))
    per_donor = {d: score(np.flatnonzero(groups == d)) for d in sorted(set(groups))}
    total = score(all_rows)  # Validates numerical values and checks overflow.
    y = np.asarray(truth.values, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            losses = [
                np.mean(np.square(y - np.asarray(a, dtype=np.float64)), axis=1)
                for a in aligned
            ]
            donor_mse = [
                float(
                    np.mean([np.mean(loss[groups == d]) for d in sorted(set(groups))])
                )
                for loss in losses
            ]
    except FloatingPointError as exc:
        raise ValueError(
            "Absolute MSE calculation overflowed; check the response scale"
        ) from exc
    if not np.isfinite(donor_mse).all():
        raise ValueError("Absolute MSE calculation produced a nonfinite value")
    return IdentitySensitivityResult(
        paired_mse_reduction=total,
        by_donor=per_donor,
        matched_mse=donor_mse[0],
        alternative_mse=donor_mse[1],
        n_units=len(truth.units),
        n_genes=len(truth.genes),
    )
