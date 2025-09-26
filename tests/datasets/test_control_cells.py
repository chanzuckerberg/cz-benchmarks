import numpy as np
import pandas as pd
import pytest
from czbenchmarks.datasets.utils_control_cells import get_matched_controls_2


def _make_obs(control_values, treatment_values, condition="pertA", group="gem1"):
    rows = []
    for cell_id, metric in control_values:
        rows.append(
            {
                "cell_id": cell_id,
                "condition": "non-targeting",
                "gem_group": group,
                "metric": metric,
            }
        )
    for cell_id, metric in treatment_values:
        rows.append(
            {
                "cell_id": cell_id,
                "condition": condition,
                "gem_group": group,
                "metric": metric,
            }
        )
    return pd.DataFrame(rows).set_index("cell_id")


@pytest.mark.parametrize(
    "control_values,treatment_values,allowed_assignments,unassigned_allowed,unassigned_size",
    [
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 5.0), ("ctrl3", 9.5)],
            [("treat1", 0.2), ("treat2", 9.8)],
            {"treat1": {"ctrl1"}, "treat2": {"ctrl3"}},
            {"ctrl2"},
            1,
            id="more_controls_no_ties",
        ),
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 1.0), ("ctrl3", 4.0)],
            [("treat1", 0.5), ("treat2", 3.8)],
            {"treat1": {"ctrl1", "ctrl2"}, "treat2": {"ctrl3"}},
            {"ctrl1", "ctrl2"},
            1,
            id="more_controls_with_ties",
        ),
    ],
)
def test_more_controls(control_values, treatment_values, allowed_assignments, unassigned_allowed, unassigned_size):
    obs = _make_obs(control_values=control_values, treatment_values=treatment_values)

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert set(assignments.keys()) == set(allowed_assignments)
    for treatment_id, allowed_controls in allowed_assignments.items():
        assert assignments[treatment_id] in allowed_controls

    assigned_controls = set(assignments.values())
    all_controls = {cell_id for cell_id, _ in control_values}
    unassigned_controls = all_controls - assigned_controls
    assert len(unassigned_controls) == unassigned_size
    assert unassigned_controls <= unassigned_allowed


@pytest.mark.parametrize(
    "control_values,treatment_values,allowed_assignments",
    [
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 5.0)],
            [("treat1", 0.1), ("treat2", 5.2)],
            {"treat1": {"ctrl1"}, "treat2": {"ctrl2"}},
            id="equal_counts_no_ties",
        ),
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 2.0)],
            [("treat1", 0.0), ("treat2", 2.0)],
            {"treat1": {"ctrl1", "ctrl2"}, "treat2": {"ctrl1", "ctrl2"}},
            id="equal_counts_with_ties",
        ),
    ],
)
def test_equal_counts(control_values, treatment_values, allowed_assignments):
    obs = _make_obs(control_values=control_values, treatment_values=treatment_values)

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert set(assignments.keys()) == set(allowed_assignments)
    for treatment_id, allowed_controls in allowed_assignments.items():
        assert assignments[treatment_id] in allowed_controls

    all_controls = {cell_id for cell_id, _ in control_values}
    assert set(assignments.values()) == all_controls


@pytest.mark.parametrize(
    "control_values,treatment_values,expected_assignments,allowed_surplus",
    [
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 10.0)],
            [("treat1", 0.1), ("treat2", 10.2), ("treat3", 20.0)],
            {"treat1": {"ctrl1"}, "treat2": {"ctrl2"}},
            {"treat3"},
            id="more_treatments_no_ties",
        ),
        pytest.param(
            [("ctrl1", 0.0), ("ctrl2", 10.0)],
            [("treat1", -0.2), ("treat2", 0.2), ("treat3", 10.0)],
            {"treat3": {"ctrl2"}, "treat1": {"ctrl1", "ctrl2"}, "treat2": {"ctrl1", "ctrl2"}},
            {"treat1", "treat2"},
            id="more_treatments_with_ties",
        ),
    ],
)
def test_more_treatments(control_values, treatment_values, expected_assignments, allowed_surplus):
    obs = _make_obs(control_values=control_values, treatment_values=treatment_values)

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert len(assignments) == len(control_values)
    for treatment_id, allowed_controls in expected_assignments.items():
        if treatment_id in assignments:
            assert assignments[treatment_id] in allowed_controls

    unassigned_treatments = {cell_id for cell_id, _ in treatment_values} - set(assignments.keys())
    expected_surplus_count = max(0, len(treatment_values) - len(control_values))
    assert len(unassigned_treatments) == expected_surplus_count
    assert unassigned_treatments <= allowed_surplus

