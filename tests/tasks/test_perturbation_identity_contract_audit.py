"""Independent adversarial checks of labeled preaggregated-response semantics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
    IdentityPredictions,
    LabeledResponses,
    compare_identity_predictions,
)


@pytest.fixture
def labeled_panel():
    units = pd.MultiIndex.from_tuples(
        [("d2", "b"), ("d1", "a"), ("d2", "a"), ("d3", "c")]
    )
    genes = pd.Index(["g3", "g1", "g2"])
    observed = np.array([[0, 1, 5], [2, 0, 9], [4, 7, 1], [9, 4, 6]], dtype=float)
    model_error = np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1], [3, 2, 0]], dtype=float)
    alternate_error = np.array(
        [[1, 3, 2], [4, 1, 1], [3, 1, 0], [0, 1, 0]], dtype=float
    )
    truth = LabeledResponses(observed, units, genes, "one-context matched NTC delta")
    matched = IdentityPredictions(
        replace(truth, values=observed + model_error), ["b", "a", "a", "c"]
    )
    alternative = IdentityPredictions(
        replace(truth, values=observed + alternate_error), ["a", "b", "b", "a"]
    )
    return truth, matched, alternative


def _reorder_response(x, rows, cols):
    return replace(
        x,
        values=x.values[np.ix_(rows, cols)],
        units=x.units[rows],
        genes=x.genes[cols],
    )


def _reorder_prediction(x, rows, cols):
    return replace(
        x,
        responses=_reorder_response(x.responses, rows, cols),
        identity_inputs=np.asarray(x.identity_inputs)[rows],
    )


def test_independent_axis_permutations_match_manual_donor_losses(labeled_panel):
    truth, matched, alternative = labeled_panel
    donor = np.asarray(truth.units.get_level_values(0))
    manual = {}
    absolute = []
    for d in sorted(set(donor)):
        mask = donor == d
        ml = float(np.mean((truth.values[mask] - matched.responses.values[mask]) ** 2))
        al = float(
            np.mean((truth.values[mask] - alternative.responses.values[mask]) ** 2)
        )
        manual[d] = al - ml
        absolute.append([ml, al])
    result = compare_identity_predictions(
        _reorder_response(truth, [1, 3, 0, 2], [1, 2, 0]),
        _reorder_prediction(matched, [3, 1, 2, 0], [2, 0, 1]),
        _reorder_prediction(alternative, [2, 0, 1, 3], [0, 2, 1]),
    )
    assert result.by_donor == pytest.approx(manual)
    assert result.paired_mse_reduction == pytest.approx(np.mean(list(manual.values())))
    assert result.matched_mse == pytest.approx(np.mean(absolute, axis=0)[0])
    assert result.alternative_mse == pytest.approx(np.mean(absolute, axis=0)[1])
    assert result.alternative_mse - result.matched_mse == pytest.approx(
        result.paired_mse_reduction
    )


def test_gene_values_without_corresponding_labels_do_not_count_as_reordering(
    labeled_panel,
):
    truth, matched, alternative = labeled_panel
    expected = compare_identity_predictions(*labeled_panel)
    broken = replace(
        matched,
        responses=replace(matched.responses, values=matched.responses.values[:, ::-1]),
    )
    observed = compare_identity_predictions(truth, broken, alternative)
    assert observed.paired_mse_reduction != pytest.approx(expected.paired_mse_reduction)


@pytest.mark.parametrize("which", ["truth", "matched", "alternative"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j])
def test_nonfinite_or_complex_values_rejected_in_every_matrix(
    labeled_panel, which, bad
):
    truth, matched, alternative = labeled_panel
    response = (
        truth
        if which == "truth"
        else (matched if which == "matched" else alternative).responses
    )
    values = response.values.astype(complex if isinstance(bad, complex) else float)
    values[1, 2] = bad
    changed = replace(response, values=values)
    if which == "truth":
        truth = changed
    elif which == "matched":
        matched = replace(matched, responses=changed)
    else:
        alternative = replace(alternative, responses=changed)
    with pytest.raises(ValueError):
        compare_identity_predictions(truth, matched, alternative)


@pytest.mark.parametrize("which", ["matched", "alternative"])
@pytest.mark.parametrize("axis", ["unit", "gene"])
@pytest.mark.parametrize("extra", [False, True])
def test_neither_subset_nor_superset_is_silently_intersected(
    labeled_panel, which, axis, extra
):
    truth, matched, alternative = labeled_panel
    pred = matched if which == "matched" else alternative
    x = pred.responses
    if axis == "unit":
        units = (
            x.units.append(pd.MultiIndex.from_tuples([("d4", "x")]))
            if extra
            else x.units[:-1]
        )
        values = np.vstack([x.values, [[1, 2, 3]]]) if extra else x.values[:-1]
        ids = [*pred.identity_inputs, "y"] if extra else pred.identity_inputs[:-1]
        pred = replace(
            pred, responses=replace(x, values=values, units=units), identity_inputs=ids
        )
    else:
        genes = x.genes.append(pd.Index(["extra"])) if extra else x.genes[:-1]
        values = np.column_stack([x.values, np.zeros(4)]) if extra else x.values[:, :-1]
        pred = replace(pred, responses=replace(x, values=values, genes=genes))
    if which == "matched":
        matched = pred
    else:
        alternative = pred
    with pytest.raises(ValueError, match="exactly the same"):
        compare_identity_predictions(truth, matched, alternative)


@pytest.mark.parametrize("which", ["matched", "alternative"])
def test_masked_identity_declarations_are_not_used_as_hidden_labels(
    labeled_panel, which
):
    truth, matched, alternative = labeled_panel
    pred = matched if which == "matched" else alternative
    pred = replace(
        pred,
        identity_inputs=np.ma.array(
            pred.identity_inputs, mask=[False, True, False, False]
        ),
    )
    if which == "matched":
        matched = pred
    else:
        alternative = pred
    with pytest.raises(ValueError, match="masked"):
        compare_identity_predictions(truth, matched, alternative)


def test_readonly_inputs_and_labels_are_not_mutated(labeled_panel):
    truth, matched, alternative = labeled_panel
    responses = [truth, matched.responses, alternative.responses]
    before = [(x.values.copy(), x.units.copy(), x.genes.copy()) for x in responses]
    for x in responses:
        x.values.flags.writeable = False
    compare_identity_predictions(*labeled_panel)
    for x, (values, units, genes) in zip(responses, before):
        np.testing.assert_array_equal(x.values, values)
        pd.testing.assert_index_equal(x.units, units)
        pd.testing.assert_index_equal(x.genes, genes)


def test_distinct_declared_inputs_can_have_identical_predictions(labeled_panel):
    truth, matched, alternative = labeled_panel
    alternative = replace(alternative, responses=matched.responses)
    result = compare_identity_predictions(truth, matched, alternative)
    assert result.paired_mse_reduction == 0
    assert all(value == 0 for value in result.by_donor.values())
    assert result.matched_mse == result.alternative_mse


def test_absolute_mse_aggregation_must_not_return_infinity():
    units = pd.MultiIndex.from_tuples([("d1", "a"), ("d1", "b")])
    genes = pd.Index(["g1"])
    truth = LabeledResponses(np.zeros((2, 1)), units, genes, "delta")
    prediction = replace(truth, values=np.full((2, 1), 1e154))
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        compare_identity_predictions(
            truth,
            IdentityPredictions(prediction, ["a", "b"]),
            IdentityPredictions(prediction, ["b", "a"]),
        )
