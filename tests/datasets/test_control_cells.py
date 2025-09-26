import numpy as np
import pandas as pd
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


def test_more_controls_no_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 5.0), ("ctrl3", 9.5)],
        treatment_values=[("treat1", 0.2), ("treat2", 9.8)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert assignments == {"treat1": "ctrl1", "treat2": "ctrl3"}
    assert set(assignments.values()) == {"ctrl1", "ctrl3"}


def test_more_controls_with_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 1.0), ("ctrl3", 4.0)],
        treatment_values=[("treat1", 0.5), ("treat2", 3.8)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert assignments["treat2"] == "ctrl3"
    assert assignments["treat1"] in {"ctrl1", "ctrl2"}
    unassigned_controls = {"ctrl1", "ctrl2", "ctrl3"} - set(assignments.values())
    assert len(unassigned_controls) == 1
    assert unassigned_controls <= {"ctrl1", "ctrl2"}


def test_equal_counts_no_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 5.0)],
        treatment_values=[("treat1", 0.1), ("treat2", 5.2)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert assignments == {"treat1": "ctrl1", "treat2": "ctrl2"}
    assert set(assignments.keys()) == {"treat1", "treat2"}


def test_equal_counts_with_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 2.0)],
        treatment_values=[("treat1", 0.0), ("treat2", 2.0)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert set(assignments.keys()) == {"treat1", "treat2"}
    assert set(assignments.values()) == {"ctrl1", "ctrl2"}


def test_more_treatments_no_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 10.0)],
        treatment_values=[("treat1", 0.1), ("treat2", 10.2), ("treat3", 20.0)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert assignments == {"treat1": "ctrl1", "treat2": "ctrl2"}
    assert "treat3" not in assignments
    assert len(assignments) == 2


def test_more_treatments_with_ties():
    obs = _make_obs(
        control_values=[("ctrl1", 0.0), ("ctrl2", 10.0)],
        treatment_values=[("treat1", -0.2), ("treat2", 0.2), ("treat3", 10.0)],
    )

    result = get_matched_controls_2(obs, ["metric"])
    assignments = dict(result["pertA"])

    assert assignments["treat3"] == "ctrl2"
    assigned_to_ctrl1 = {t for t, c in assignments.items() if c == "ctrl1"}
    assert assigned_to_ctrl1 in [{"treat1"}, {"treat2"}]
    unassigned_treatments = {"treat1", "treat2", "treat3"} - set(assignments.keys())
    assert len(unassigned_treatments) == 1
    assert unassigned_treatments <= {"treat1", "treat2"}

