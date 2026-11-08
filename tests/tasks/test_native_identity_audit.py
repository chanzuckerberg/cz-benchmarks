"""Independent audit of the registered response task, not biological validation."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.metrics.utils import aggregate_results
from czbenchmarks.tasks import (
    TASK_REGISTRY,
    PerturbationIdentityTask,
    PerturbationIdentityTaskInput,
)


@pytest.fixture
def native_panel():
    from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
        IdentityPredictions,
        LabeledResponses,
    )

    units = pd.MultiIndex.from_tuples(
        [("d2", "b"), ("d1", "a"), ("d2", "a"), ("d3", "c")]
    )
    genes = pd.Index(["g3", "g1", "g2"])
    truth = np.array([[0, 1, 5], [2, 0, 9], [4, 7, 1], [9, 4, 6]], dtype=float)
    matched = truth + np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1], [3, 2, 0]])
    alternative = truth + np.array([[1, 3, 2], [4, 1, 1], [3, 1, 0], [0, 1, 0]])
    generic = np.zeros_like(truth)
    labels = LabeledResponses(truth, units, genes, "prespecified response units")
    fitting = {d: [x for x in ("d1", "d2", "d3") if x != d] for d in ("d1", "d2", "d3")}
    data = PerturbationIdentityTaskInput(
        truth=labels,
        matched_units=units,
        matched_genes=genes,
        matched_scale=labels.scale,
        matched_identity_inputs=["b", "a", "a", "c"],
        alternative=IdentityPredictions(
            replace(labels, values=alternative), ["a", "b", "b", "a"]
        ),
        generic=replace(labels, values=generic),
        fitted_donors={
            "matched": deepcopy(fitting),
            "alternative": deepcopy(fitting),
            "generic": deepcopy(fitting),
            "learned_preprocessing": {d: [] for d in fitting},
        },
        dataset_id="independent_audit_fixture",
        context="single_context",
        matched_model_id="fixed_synthetic_predictor",
        alternative_reference_id="fixed_synthetic_alternative",
        generic_reference_id="reference_a",
        heldout_control_usage="none at inference",
        alternative_kind="model_input_rerun",
        alternative_provenance="Fixed-model alternative target inputs; synthetic test.",
        generic_provenance="Training-only constant reference A.",
    )
    return matched, data


def _measures(results, donor=None):
    return {
        r.params["measure"]: r.value for r in results if r.params.get("donor") == donor
    }


def test_registry_validation_native_run_and_json_roundtrip(native_panel):
    matched, data = native_panel
    key = "perturbation_identity_response"
    assert key in TASK_REGISTRY.list_tasks()
    assert TASK_REGISTRY.get_task_class(key) is PerturbationIdentityTask
    info = TASK_REGISTRY.get_task_info(key)
    assert info.name == "PerturbationIdentityTask"
    assert info.description != "Task introspection failed."
    built = TASK_REGISTRY.validate_task_inputs(
        key, {name: getattr(data, name) for name in type(data).model_fields}
    )
    results = TASK_REGISTRY.get_task_class(key)().run(matched, built)
    assert len(results) == 20
    assert len({r.aggregation_key for r in results}) == len(results)
    assert {r.metric_type for r in results} == {
        MetricType.MEAN_SQUARED_ERROR,
        MetricType.PAIRED_MSE_REDUCTION,
    }
    assert {r.params["measure"] for r in results} == {
        "matched_mse",
        "alternative_mse",
        "generic_mse",
        "alternative_minus_matched_mse",
        "generic_minus_matched_mse",
    }
    recovered = [
        MetricResult(**value)
        for value in json.loads(
            json.dumps([r.model_dump(mode="json") for r in results])
        )
    ]
    assert [r.aggregation_key for r in recovered] == [
        r.aggregation_key for r in results
    ]
    aggregated = aggregate_results(results + results)
    assert len(aggregated) == len(results)
    assert all(r.n_values == 2 for r in aggregated)


def test_native_task_matches_manual_donor_balanced_losses(native_panel):
    matched, data = native_panel
    results = PerturbationIdentityTask().run(matched, data)
    donors = np.asarray(data.truth.units.get_level_values(0))
    manual = []
    for donor in sorted(set(donors)):
        ix = donors == donor
        true = data.truth.values[ix]
        losses = {
            "matched_mse": np.mean((true - matched[ix]) ** 2),
            "alternative_mse": np.mean(
                (true - data.alternative.responses.values[ix]) ** 2
            ),
            "generic_mse": np.mean((true - data.generic.values[ix]) ** 2),
        }
        losses["alternative_minus_matched_mse"] = (
            losses["alternative_mse"] - losses["matched_mse"]
        )
        losses["generic_minus_matched_mse"] = (
            losses["generic_mse"] - losses["matched_mse"]
        )
        assert _measures(results, donor) == pytest.approx(losses)
        manual.append(losses)
    macro = {name: np.mean([row[name] for row in manual]) for name in manual[0]}
    assert _measures(results) == pytest.approx(macro)


def test_every_input_has_independent_row_and_gene_order(native_panel):
    matched, data = native_panel

    def reorder(x, rows, cols):
        return replace(
            x,
            values=x.values[np.ix_(rows, cols)],
            units=x.units[rows],
            genes=x.genes[cols],
        )

    truth = reorder(data.truth, [1, 3, 0, 2], [1, 2, 0])
    mrows, mcols = [3, 1, 2, 0], [2, 0, 1]
    arows, acols = [2, 0, 1, 3], [0, 2, 1]
    alt = replace(
        data.alternative,
        responses=reorder(data.alternative.responses, arows, acols),
        identity_inputs=np.asarray(data.alternative.identity_inputs)[arows].tolist(),
    )
    generic = reorder(data.generic, [3, 2, 1, 0], [2, 1, 0])
    permuted = data.model_copy(
        update={
            "truth": truth,
            "matched_units": data.matched_units[mrows],
            "matched_genes": data.matched_genes[mcols],
            "matched_identity_inputs": np.asarray(data.matched_identity_inputs)[
                mrows
            ].tolist(),
            "alternative": alt,
            "generic": generic,
        }
    )
    actual = PerturbationIdentityTask().run(matched[np.ix_(mrows, mcols)], permuted)
    expected = PerturbationIdentityTask().run(matched, data)
    assert {r.aggregation_key: r.value for r in actual} == pytest.approx(
        {r.aggregation_key: r.value for r in expected}
    )


@pytest.mark.parametrize(
    "component", ["matched", "alternative", "generic", "learned_preprocessing"]
)
def test_declared_heldout_donor_leakage_rejected_for_each_component(
    native_panel, component
):
    matched, data = native_panel
    fitting = deepcopy(data.fitted_donors)
    fitting[component]["d1"].append("d1")
    with pytest.raises(ValueError, match="held-out donor"):
        PerturbationIdentityTask().run(
            matched, data.model_copy(update={"fitted_donors": fitting})
        )


@pytest.mark.parametrize("kind", ["model_input_rerun", "output_reassignment"])
def test_alternative_cannot_declare_a_different_fitting_cohort(native_panel, kind):
    matched, data = native_panel
    fitting = deepcopy(data.fitted_donors)
    fitting["alternative"]["d1"] = ["some_other_training_donor"]
    with pytest.raises(ValueError, match="identical fitted donors"):
        PerturbationIdentityTask().run(
            matched,
            data.model_copy(
                update={"fitted_donors": fitting, "alternative_kind": kind}
            ),
        )


@pytest.mark.parametrize("representation", ["frame", "csr"])
def test_unsupported_native_representations_are_rejected_clearly(
    native_panel, representation
):
    matched, data = native_panel
    value = (
        pd.DataFrame(matched)
        if representation == "frame"
        else sparse.csr_matrix(matched)
    )
    with pytest.raises(ValueError, match="dense numpy response matrix"):
        PerturbationIdentityTask().run(value, data)


def test_does_not_inherit_expression_pca_baseline(native_panel):
    matched, _ = native_panel
    with pytest.raises(NotImplementedError, match="generic reference"):
        PerturbationIdentityTask().compute_baseline(matched)


def test_target_varying_generic_is_rejected(native_panel):
    matched, data = native_panel
    bad = data.generic.values.copy()
    bad[0, 0] = 1
    with pytest.raises(ValueError, match="identical across targets"):
        PerturbationIdentityTask().run(
            matched,
            data.model_copy(update={"generic": replace(data.generic, values=bad)}),
        )


def test_repeated_task_runs_do_not_reuse_prior_output(native_panel):
    matched, data = native_panel
    task = PerturbationIdentityTask()
    first = task.run(matched, data)
    second = task.run(data.truth.values.copy(), data)
    assert _measures(second)["matched_mse"] == 0
    assert _measures(first)["matched_mse"] > 0
    assert _measures(first)["generic_mse"] == _measures(second)["generic_mse"]


def test_incompatible_response_scales_do_not_share_aggregation_keys(native_panel):
    matched, data = native_panel
    different_scale = "a_different_squared_unit"
    other = data.model_copy(
        update={
            "truth": replace(data.truth, scale=different_scale),
            "matched_scale": different_scale,
            "alternative": replace(
                data.alternative,
                responses=replace(data.alternative.responses, scale=different_scale),
            ),
            "generic": replace(data.generic, scale=different_scale),
        }
    )
    first = PerturbationIdentityTask().run(matched, data)
    second = PerturbationIdentityTask().run(matched, other)
    assert {r.aggregation_key for r in first}.isdisjoint(
        {r.aggregation_key for r in second}
    )


def test_distinct_generic_references_remain_distinct_when_aggregated(native_panel):
    matched, data = native_panel
    other = data.model_copy(
        update={
            "generic_reference_id": "reference_b",
            "generic_provenance": "Different training-only constant reference B.",
        }
    )
    first = PerturbationIdentityTask().run(matched, data)
    second = PerturbationIdentityTask().run(matched, other)
    keys_a = {
        r.aggregation_key for r in first if r.params["measure"].startswith("generic")
    }
    keys_b = {
        r.aggregation_key for r in second if r.params["measure"].startswith("generic")
    }
    assert keys_a.isdisjoint(keys_b)
