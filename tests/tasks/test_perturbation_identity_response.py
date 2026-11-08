import numpy as np
import pandas as pd
import pytest

from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
    IdentityPredictions,
    LabeledResponses,
    compare_identity_predictions,
)


def make_data():
    # One donor has two targets and the other just one: the average is donor-weighted.
    units = pd.MultiIndex.from_tuples([("d1", "a"), ("d1", "b"), ("d2", "a")])
    genes = pd.Index(["g1", "g2"])
    truth = LabeledResponses(np.zeros((3, 2)), units, genes, "delta-log1p-cp10k")
    matched = IdentityPredictions(
        LabeledResponses(
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]), units, genes, truth.scale
        ),
        ["a", "b", "a"],
    )
    alternative = IdentityPredictions(
        LabeledResponses(
            np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]]), units, genes, truth.scale
        ),
        ["b", "a", "b"],
    )
    return truth, matched, alternative


def test_donor_weighting_and_independent_manual_calculation():
    result = compare_identity_predictions(*make_data())
    assert result.by_donor == {"d1": 1.0, "d2": 8.0}
    assert result.paired_mse_reduction == 4.5
    assert result.matched_mse == 0.5
    assert result.alternative_mse == 5.0
    assert (result.n_units, result.n_genes) == (3, 2)


def test_reorders_both_rows_and_genes_for_each_prediction():
    truth, matched, alternative = make_data()
    truth = LabeledResponses(
        np.array([[1.0, 7.0], [2.0, -3.0], [5.0, 9.0]]),
        truth.units,
        truth.genes,
        truth.scale,
    )
    matched = IdentityPredictions(
        LabeledResponses(
            np.array([[1.1, 8.0], [0.0, -2.0], [4.0, 12.0]]),
            truth.units,
            truth.genes,
            truth.scale,
        ),
        matched.identity_inputs,
    )
    alternative = IdentityPredictions(
        LabeledResponses(
            np.array([[4.0, 2.0], [2.0, -5.0], [8.0, 3.0]]),
            truth.units,
            truth.genes,
            truth.scale,
        ),
        alternative.identity_inputs,
    )
    order, cols = [2, 0, 1], [1, 0]

    def reorder(p):
        x = p.responses
        return IdentityPredictions(
            LabeledResponses(
                x.values[np.ix_(order, cols)], x.units[order], x.genes[cols], x.scale
            ),
            np.array(p.identity_inputs)[order],
        )

    expected = compare_identity_predictions(truth, matched, alternative)
    assert (
        compare_identity_predictions(truth, reorder(matched), alternative) == expected
    )
    assert (
        compare_identity_predictions(truth, matched, reorder(alternative)) == expected
    )
    assert (
        compare_identity_predictions(truth, reorder(matched), reorder(alternative))
        == expected
    )


@pytest.mark.parametrize(
    "kind",
    [
        "unit",
        "gene",
        "duplicate-unit",
        "duplicate-gene",
        "scale",
        "matched-id",
        "alternative-id",
        "masked-id",
        "shape",
        "masked",
        "nan",
    ],
)
def test_mismatched_or_invalid_data_are_rejected(kind):
    truth, matched, alternative = make_data()
    x = alternative.responses
    values, units, genes, scale, ids = (
        x.values.copy(),
        x.units,
        x.genes,
        x.scale,
        list(alternative.identity_inputs),
    )
    if kind == "unit":
        units = pd.MultiIndex.from_tuples([("d1", "a"), ("d1", "b"), ("d3", "a")])
    if kind == "gene":
        genes = pd.Index(["g1", "g3"])
    if kind == "duplicate-unit":
        units = pd.MultiIndex.from_tuples([("d1", "a"), ("d1", "a"), ("d2", "a")])
    if kind == "duplicate-gene":
        genes = pd.Index(["g1", "g1"])
    if kind == "scale":
        scale = "raw-count"
    if kind == "matched-id":
        matched = IdentityPredictions(matched.responses, ["b", "b", "a"])
    if kind == "alternative-id":
        ids[0] = "a"
    if kind == "masked-id":
        ids = np.ma.array(ids, mask=[True, False, False])
    if kind == "shape":
        values = values[:, :1]
    if kind == "masked":
        values = np.ma.array(
            values, mask=[[True, False], [False, False], [False, False]]
        )
    if kind == "nan":
        values[0, 0] = np.nan
    alternative = IdentityPredictions(
        LabeledResponses(values, units, genes, scale), ids
    )
    with pytest.raises(ValueError):
        compare_identity_predictions(truth, matched, alternative)
