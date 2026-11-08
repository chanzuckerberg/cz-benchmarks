"""Paired predictive comparisons for aligned perturbation response matrices."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def paired_mse_reduction(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_reference: npt.ArrayLike,
    group_ids: Sequence[str] | None = None,
) -> float:
    """Return reference MSE minus model MSE on the same rows and genes.

    Rows are pre-aggregated evaluation units, such as donor/perturbation pairs;
    columns are the same ordered gene panel on the same response scale. Genes
    receive equal weight. By default rows receive equal weight. If ``group_ids``
    is supplied, average rows within each group, then weight groups equally.
    For example, donor IDs prevent donors with more perturbations dominating.

    Positive values favor ``y_pred``. The result is in squared response units
    and can be negative. It is a paired predictive comparison, not a p-value,
    causal effect, or standalone measure of perturbation identity use.

    An identity-agnostic reference must be constructed without held-out outcome
    leakage. Alternatively the reference may contain predictions recomputed
    with matched alternative identities. Swapping existing response vectors
    tests label-response alignment, not whether a model uses an identity input.
    The caller owns matching, preprocessing, splitting, and uncertainty.

    Args:
        y_true: Finite real matrix, shape (n_units, n_genes).
        y_pred: Predictions with exactly the same shape, row and gene order.
        y_reference: Reference predictions with exactly the same alignment.
        group_ids: Optional nonempty string label per row for equal-group
            aggregation. Supplying donor IDs does not verify donor holdout.

    Raises:
        ValueError: Inputs are empty, masked, nonnumeric, nonfinite, mis-shaped, or
            overflow during loss calculation; or group IDs are invalid.
    """
    matrices = []
    for name, value in (
        ("y_true", y_true),
        ("y_pred", y_pred),
        ("y_reference", y_reference),
    ):
        if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
            raise ValueError(f"{name} must not contain masked values")
        try:
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must contain real values")
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a real numeric matrix") from exc
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        matrices.append(array)

    truth, prediction, reference = matrices
    if prediction.shape != truth.shape or reference.shape != truth.shape:
        raise ValueError("All matrices must have the same shape and alignment")

    groups = None
    if group_ids is not None:
        if np.ma.isMaskedArray(group_ids) and np.ma.getmaskarray(group_ids).any():
            raise ValueError("group_ids must not contain masked values")
        groups = np.asarray(group_ids, dtype=object)
        if groups.ndim != 1 or len(groups) != len(truth):
            raise ValueError("group_ids must contain exactly one label per row")
        if any(not isinstance(g, str) or not g.strip() for g in groups):
            raise ValueError("group_ids must contain nonempty string labels")

    try:
        with np.errstate(over="raise", invalid="raise"):
            model_loss = np.mean(np.square(truth - prediction), axis=1)
            reference_loss = np.mean(np.square(truth - reference), axis=1)
            row_reductions = reference_loss - model_loss
            if groups is None:
                result = np.mean(row_reductions)
            else:
                result = np.mean(
                    [
                        np.mean(row_reductions[groups == group])
                        for group in sorted(set(groups))
                    ]
                )
    except FloatingPointError as exc:
        raise ValueError(
            "Loss calculation overflowed; check the response scale"
        ) from exc
    if not np.isfinite(result):
        raise ValueError("Loss calculation produced a nonfinite value")
    return float(result)
