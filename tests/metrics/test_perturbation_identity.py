"""Tests for paired predictive comparisons, not biological model validation."""

import numpy as np
import pytest

from czbenchmarks.metrics import metrics_registry
from czbenchmarks.metrics.perturbation_identity import paired_mse_reduction
from czbenchmarks.metrics.types import MetricResult, MetricType


def test_perfect_model_against_generic_reference():
    truth = np.eye(2)
    generic = np.full((2, 2), 0.5)
    assert paired_mse_reduction(truth, truth, generic) == pytest.approx(0.25)


def test_identical_model_and_reference_have_zero_gain():
    truth = np.eye(2)
    generic = np.full((2, 2), 0.5)
    assert paired_mse_reduction(truth, generic, generic) == 0.0


def test_model_worse_than_reference_has_negative_gain():
    truth = np.eye(2)
    generic = np.full((2, 2), 0.5)
    assert paired_mse_reduction(truth, generic, truth) == pytest.approx(-0.25)


def test_matched_wrong_response_reference():
    truth = np.eye(2)
    assert paired_mse_reduction(truth, truth, truth[::-1]) == 1.0


def test_equal_group_weighting_prevents_row_count_dominance():
    truth = np.zeros((4, 1))
    reference = [[2.0], [1.0], [1.0], [1.0]]
    assert paired_mse_reduction(truth, truth, reference) == 1.75
    assert (
        paired_mse_reduction(truth, truth, reference, group_ids=["a", "b", "b", "b"])
        == 2.5
    )


def test_joint_permutation_of_rows_and_columns_preserves_score():
    rng = np.random.default_rng(10)
    truth, pred, ref = rng.normal(size=(3, 5, 4))
    groups = np.array(["a", "a", "b", "b", "b"])
    expected = paired_mse_reduction(truth, pred, ref, groups)
    order = [4, 2, 0, 3, 1]
    observed = paired_mse_reduction(
        truth[order, ::-1], pred[order, ::-1], ref[order, ::-1], groups[order]
    )
    assert observed == pytest.approx(expected)


def test_positive_rescaling_has_squared_units():
    truth = np.eye(2)
    ref = np.full((2, 2), 0.5)
    assert paired_mse_reduction(3 * truth, 3 * truth, 3 * ref) == 2.25


def test_registry_and_result_type_integration():
    value = metrics_registry.compute(
        MetricType.PAIRED_MSE_REDUCTION,
        y_true=np.eye(2),
        y_pred=np.eye(2),
        y_reference=np.zeros((2, 2)),
        group_ids=["donor_a", "donor_b"],
    )
    result = MetricResult(metric_type=MetricType.PAIRED_MSE_REDUCTION, value=value)
    assert result.value == 0.5
    assert result.metric_type.value == "paired_mse_reduction"
    assert MetricType.PAIRED_MSE_REDUCTION in metrics_registry.list_metrics(
        tags={"paired_comparison"}
    )


def test_registry_rejects_missing_reference():
    with pytest.raises(ValueError, match="Missing required arguments"):
        metrics_registry.compute(
            MetricType.PAIRED_MSE_REDUCTION, y_true=[[0]], y_pred=[[0]]
        )


@pytest.mark.parametrize(
    "bad",
    [[], [1, 2], np.empty((0, 2)), [[np.nan]], [[np.inf]], [["x"]], [[1j]]],
)
@pytest.mark.parametrize("argument", ["y_true", "y_pred", "y_reference"])
def test_reject_invalid_matrices(bad, argument):
    values = {"y_true": [[0]], "y_pred": [[0]], "y_reference": [[0]]}
    values[argument] = bad
    with pytest.raises(ValueError):
        paired_mse_reduction(**values)


def test_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        paired_mse_reduction([[0, 0]], [[0]], [[0, 0]])


@pytest.mark.parametrize("groups", [[], ["a", "b"], [["a"]], [None], [""], [1]])
def test_reject_invalid_groups(groups):
    with pytest.raises(ValueError, match="group_ids"):
        paired_mse_reduction([[0]], [[0]], [[0]], group_ids=groups)


@pytest.mark.parametrize("through_registry", [False, True])
def test_reject_masked_group_labels(through_registry):
    values = {
        "y_true": np.zeros((3, 1)),
        "y_pred": np.zeros((3, 1)),
        "y_reference": [[2.0], [1.0], [1.0]],
        "group_ids": np.ma.array(["a", "b", "b"], mask=[True, False, False]),
    }
    with pytest.raises(ValueError, match="group_ids must not contain masked values"):
        if through_registry:
            metrics_registry.compute(MetricType.PAIRED_MSE_REDUCTION, **values)
        else:
            paired_mse_reduction(**values)


def test_unmasked_group_array_preserves_group_weighting():
    groups = np.ma.array(["a", "b", "b"], mask=False)
    assert (
        paired_mse_reduction(
            np.zeros((3, 1)), np.zeros((3, 1)), [[2.0], [1.0], [1.0]], groups
        )
        == 2.5
    )


def test_reject_loss_overflow():
    with pytest.raises(ValueError, match="overflowed"):
        paired_mse_reduction([[1e308]], [[-1e308]], [[0]])


@pytest.mark.parametrize("argument", ["y_true", "y_pred", "y_reference"])
def test_reject_values_outside_float64_range(argument):
    values = {"y_true": [[0]], "y_pred": [[0]], "y_reference": [[0]]}
    values[argument] = [[10**1000]]
    with pytest.raises(ValueError, match=f"{argument} must be a real numeric matrix"):
        paired_mse_reduction(**values)


@pytest.mark.parametrize("argument", ["y_true", "y_pred", "y_reference"])
def test_reject_masked_values(argument):
    values = {"y_true": [[0]], "y_pred": [[0]], "y_reference": [[0]]}
    values[argument] = np.ma.array([[1.0]], mask=[[True]])
    with pytest.raises(ValueError, match=f"{argument} must not contain masked values"):
        paired_mse_reduction(**values)


def test_does_not_mutate_inputs():
    truth, pred, ref = np.eye(3), np.zeros((3, 3)), np.ones((3, 3))
    before = [v.copy() for v in (truth, pred, ref)]
    paired_mse_reduction(truth, pred, ref)
    for original, current in zip(before, (truth, pred, ref)):
        np.testing.assert_array_equal(original, current)
