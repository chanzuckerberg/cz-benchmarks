"""Native task, registry and donor-split contract checks."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from czbenchmarks.metrics.types import MetricType
from czbenchmarks.tasks import TASK_REGISTRY, PerturbationIdentityTask
from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
    IdentityPredictions,
    LabeledResponses,
)


@pytest.fixture
def case():
    units = pd.MultiIndex.from_tuples(
        [("d1", "a"), ("d1", "b"), ("d2", "a"), ("d2", "b")]
    )
    genes = pd.Index(["g1", "g2"])
    truth_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    matched = truth_values + np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    alternative = truth_values + 1.0
    generic = np.array([[2.0, 2.0], [2.0, 2.0], [6.0, 6.0], [6.0, 6.0]])
    scale = "treated minus matched control, log normalized"
    truth = LabeledResponses(truth_values, units, genes, scale)
    alternative = IdentityPredictions(
        LabeledResponses(alternative, units, genes, scale), ["b", "a", "b", "a"]
    )
    generic = LabeledResponses(generic, units, genes, scale)
    folds = {"d1": ["d2"], "d2": ["d1"]}
    manifest = {
        k: deepcopy(folds)
        for k in ["matched", "alternative", "generic", "learned_preprocessing"]
    }
    params = dict(
        truth=truth,
        matched_units=units,
        matched_genes=genes,
        matched_scale=scale,
        matched_identity_inputs=["a", "b", "a", "b"],
        alternative=alternative,
        generic=generic,
        fitted_donors=manifest,
        dataset_id="fixture",
        context="one cell type and condition",
        matched_model_id="model",
        alternative_reference_id="cyclic",
        generic_reference_id="train-mean",
        heldout_control_usage="truth uses held-out donor control only",
        alternative_kind="model_input_rerun",
        alternative_provenance="test identity inputs rerun",
        generic_provenance="pooled training-only response mean",
    )
    return matched, params


def metric_map(results):
    return {(r.params["measure"], r.params.get("donor")): r.value for r in results}


def test_native_registry_run_and_metric_serialization(case):
    matched, params = case
    task_name = "perturbation_identity_response"
    assert task_name in TASK_REGISTRY.list_tasks()
    cls = TASK_REGISTRY.get_task_class(task_name)
    assert cls is PerturbationIdentityTask
    inputs = TASK_REGISTRY.validate_task_inputs(task_name, params)
    task = cls()
    results = task.run(matched, inputs)
    assert len(results) == 15
    assert len({r.aggregation_key for r in results}) == len(results)
    assert all(isinstance(r.model_dump_json(), str) for r in results)
    observed = metric_map(results)
    expected_matched = np.mean((params["truth"].values - matched) ** 2)
    expected_generic = np.mean((params["truth"].values - params["generic"].values) ** 2)
    assert observed[("matched_mse", None)] == pytest.approx(expected_matched)
    assert observed[("alternative_mse", None)] == pytest.approx(1.0)
    assert observed[("generic_mse", None)] == pytest.approx(expected_generic)
    assert observed[("alternative_minus_matched_mse", None)] == pytest.approx(
        1.0 - expected_matched
    )
    assert observed[("generic_minus_matched_mse", None)] == pytest.approx(
        expected_generic - expected_matched
    )
    assert {r.metric_type for r in results} == {
        MetricType.MEAN_SQUARED_ERROR,
        MetricType.PAIRED_MSE_REDUCTION,
    }
    assert all(
        r.params["alternative_kind"] == "model_input_rerun"
        for r in results
        if r.params["measure"].startswith("alternative")
    )
    with pytest.raises(NotImplementedError):
        task.compute_baseline(matched)
    # Repeated use of a Task instance does not retain stale predictions.
    second = metric_map(task.run(params["truth"].values, inputs))
    assert second[("matched_mse", None)] == 0.0
    assert second[("generic_minus_matched_mse", None)] == pytest.approx(
        expected_generic
    )


@pytest.mark.parametrize(
    "component", ["matched", "alternative", "generic", "learned_preprocessing"]
)
def test_declared_heldout_donor_in_any_fitted_component_is_rejected(case, component):
    matched, params = case
    params["fitted_donors"][component]["d1"].append("d1")
    with pytest.raises(ValueError, match="held-out donor"):
        PerturbationIdentityTask().run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )


def test_generic_must_be_target_invariant_within_donor(case):
    matched, params = case
    x = params["generic"].values.copy()
    x[1, 0] += 1
    params["generic"] = replace(params["generic"], values=x)
    with pytest.raises(ValueError, match="identical across targets"):
        PerturbationIdentityTask().run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )


@pytest.mark.parametrize("matrix", ["dataframe", "sparse"])
def test_non_dense_input_rejected_without_discarding_labels(case, matrix):
    matched, params = case
    bad = pd.DataFrame(matched) if matrix == "dataframe" else csr_matrix(matched)
    with pytest.raises(ValueError, match="dense numpy"):
        PerturbationIdentityTask().run(
            bad,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )


@pytest.mark.parametrize(
    "bad",
    [
        "missing_component",
        "missing_donor",
        "duplicate",
        "rerun_train_mismatch",
        "blank_context",
        "mismatched_scale",
        "mismatched_genes",
        "alt_identity",
    ],
)
def test_invalid_contract_rejected(case, bad):
    matched, params = case
    if bad == "missing_component":
        params["fitted_donors"].pop("generic")
    if bad == "missing_donor":
        params["fitted_donors"]["generic"].pop("d1")
    if bad == "duplicate":
        params["fitted_donors"]["generic"]["d1"] = ["d2", "d2"]
    if bad == "rerun_train_mismatch":
        params["fitted_donors"]["alternative"]["d1"] = ["d3"]
    if bad == "blank_context":
        params["context"] = "  "
    if bad == "mismatched_scale":
        params["generic"] = replace(params["generic"], scale="different")
    if bad == "mismatched_genes":
        params["generic"] = replace(params["generic"], genes=pd.Index(["g1", "other"]))
    if bad == "alt_identity":
        params["alternative"] = replace(
            params["alternative"], identity_inputs=["a", "a", "b", "a"]
        )
    with pytest.raises(ValueError):
        PerturbationIdentityTask().run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )


def test_independently_reordered_axes_and_output_reassignment_provenance(case):
    matched, params = case
    task = PerturbationIdentityTask()
    expected = metric_map(
        task.run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )
    )
    rows, cols = [3, 1, 0, 2], [1, 0]
    matched = matched[np.ix_(rows, cols)]
    params["matched_units"] = params["matched_units"][rows]
    params["matched_genes"] = params["matched_genes"][cols]
    params["matched_identity_inputs"] = np.asarray(params["matched_identity_inputs"])[
        rows
    ].tolist()
    other_rows, other_cols = [2, 3, 0, 1], [1, 0]
    for name in ["alternative", "generic"]:
        obj = params[name]
        x = obj.responses if name == "alternative" else obj
        x = replace(
            x,
            values=x.values[np.ix_(other_rows, other_cols)],
            units=x.units[other_rows],
            genes=x.genes[other_cols],
        )
        params[name] = (
            replace(
                obj,
                responses=x,
                identity_inputs=np.asarray(obj.identity_inputs)[other_rows].tolist(),
            )
            if name == "alternative"
            else x
        )
    params["alternative_kind"] = "output_reassignment"
    params["alternative_provenance"] = "explicit output reassignment control"
    results = task.run(
        matched,
        TASK_REGISTRY.validate_task_inputs("perturbation_identity_response", params),
    )
    assert metric_map(results) == pytest.approx(expected)
    assert all(
        r.params["alternative_kind"] == "output_reassignment"
        for r in results
        if r.params["measure"].startswith("alternative")
    )


def test_donor_weighting_with_unequal_target_counts(case):
    matched, params = case
    rows = [0, 1, 2]
    matched = matched[rows]
    params["truth"] = replace(
        params["truth"],
        values=params["truth"].values[rows],
        units=params["truth"].units[rows],
    )
    params["matched_units"] = params["matched_units"][rows]
    params["matched_identity_inputs"] = [
        params["matched_identity_inputs"][i] for i in rows
    ]
    alt = params["alternative"]
    params["alternative"] = replace(
        alt,
        responses=replace(
            alt.responses,
            values=alt.responses.values[rows],
            units=alt.responses.units[rows],
        ),
        identity_inputs=[alt.identity_inputs[i] for i in rows],
    )
    x = params["generic"]
    params["generic"] = replace(x, values=x.values[rows], units=x.units[rows])
    results = metric_map(
        PerturbationIdentityTask().run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )
    )
    donor1 = np.mean((params["truth"].values[:2] - matched[:2]) ** 2)
    donor2 = np.mean((params["truth"].values[2:] - matched[2:]) ** 2)
    assert results[("matched_mse", None)] == pytest.approx((donor1 + donor2) / 2)
    assert results[("matched_mse", None)] != pytest.approx(
        np.mean((params["truth"].values - matched) ** 2)
    )


def test_metric_keys_distinguish_references_scale_and_panel(case):
    matched, params = case
    task = PerturbationIdentityTask()

    def keys(p):
        return {
            (r.params["measure"], r.params.get("donor")): r.aggregation_key
            for r in task.run(
                matched,
                TASK_REGISTRY.validate_task_inputs("perturbation_identity_response", p),
            )
        }

    baseline = keys(params)
    changed = deepcopy(params)
    changed["generic_reference_id"] = "different-reference"
    other = keys(changed)
    assert (
        other[("generic_minus_matched_mse", None)]
        != baseline[("generic_minus_matched_mse", None)]
    )
    assert (
        other[("alternative_minus_matched_mse", None)]
        == baseline[("alternative_minus_matched_mse", None)]
    )
    changed = deepcopy(params)
    changed["alternative_reference_id"] = "different-alternative"
    other = keys(changed)
    assert (
        other[("alternative_minus_matched_mse", None)]
        != baseline[("alternative_minus_matched_mse", None)]
    )
    changed = deepcopy(params)
    changed["truth"] = replace(changed["truth"], scale="different")
    changed["matched_scale"] = "different"
    changed["generic"] = replace(changed["generic"], scale="different")
    changed["alternative"] = replace(
        changed["alternative"],
        responses=replace(changed["alternative"].responses, scale="different"),
    )
    assert keys(changed)[("matched_mse", None)] != baseline[("matched_mse", None)]


def test_output_reassignment_also_requires_same_declared_fit_donors(case):
    matched, params = case
    params["alternative_kind"] = "output_reassignment"
    params["fitted_donors"]["alternative"]["d1"] = ["different"]
    with pytest.raises(ValueError, match="identical fitted donors"):
        PerturbationIdentityTask().run(
            matched,
            TASK_REGISTRY.validate_task_inputs(
                "perturbation_identity_response", params
            ),
        )
