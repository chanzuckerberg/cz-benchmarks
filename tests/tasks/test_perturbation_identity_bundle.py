"""Portable-bundle schema, integrity and native score regression tests."""

from dataclasses import replace
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from czbenchmarks.tasks import TASK_REGISTRY
from czbenchmarks.tasks.single_cell.perturbation_identity_bundle import (
    ARRAY_NAMES,
    load_prediction_bundle,
    main,
    save_prediction_bundle,
)
from tests.tasks import test_perturbation_identity_task as task_tests


@pytest.fixture
def case():
    return task_tests.case.__wrapped__()


@pytest.fixture
def bundle_case(case, tmp_path):
    matched, params = case
    inputs = TASK_REGISTRY.validate_task_inputs(
        "perturbation_identity_response", params
    )
    path = tmp_path / "input.npz"
    digest = save_prediction_bundle(path, matched, inputs)
    return path, digest, matched, inputs


def rewrite(
    path, *, array_updates=None, manifest_update=None, refresh=True, extra=None
):
    with zipfile.ZipFile(path) as archive:
        raw = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(
        np.load(io.BytesIO(raw["manifest.npy"]), allow_pickle=False).item()
    )
    for key, value in (array_updates or {}).items():
        b = io.BytesIO()
        np.save(b, value)
        raw[key + ".npy"] = b.getvalue()
        if refresh:
            manifest["array_sha256"][key] = hashlib.sha256(b.getvalue()).hexdigest()
    if manifest_update:
        manifest_update(manifest)
    b = io.BytesIO()
    np.save(b, np.asarray(json.dumps(manifest)), allow_pickle=False)
    raw["manifest.npy"] = b.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in raw.items():
            archive.writestr(name, value)
        if extra:
            archive.writestr(extra, b"bad")


def test_deterministic_round_trip_and_cli(bundle_case, tmp_path):
    path, digest, matched, inputs = bundle_case
    loaded = load_prediction_bundle(path, expected_sha256=digest)
    assert loaded.sha256 == digest
    np.testing.assert_array_equal(loaded.matched_values, matched)
    assert loaded.task_input.fitted_donors == inputs.fitted_donors
    assert len(loaded.score()) == 15
    other = tmp_path / "other.npz"
    assert save_prediction_bundle(other, matched, inputs) == digest
    assert path.read_bytes() == other.read_bytes()
    output = tmp_path / "results.json"
    assert main([str(path), "--output", str(output), "--expected-sha256", digest]) == 0
    result = json.loads(output.read_text())
    assert result["bundle_sha256"] == digest
    assert len(result["metrics"]) == 15
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == ARRAY_NAMES | {"manifest"}
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_independent_axis_orders_preserved(case, tmp_path):
    matched, params = case
    row = np.array([3, 0, 2, 1])
    col = np.array([1, 0])
    alt = params["alternative"]
    response = alt.responses
    params["alternative"] = replace(
        alt,
        responses=replace(
            response,
            values=response.values[np.ix_(row, col)],
            units=response.units[row],
            genes=response.genes[col],
        ),
        identity_inputs=np.asarray(alt.identity_inputs)[row].tolist(),
    )
    inputs = TASK_REGISTRY.validate_task_inputs(
        "perturbation_identity_response", params
    )
    path = tmp_path / "permuted.npz"
    save_prediction_bundle(path, matched, inputs)
    loaded = load_prediction_bundle(path)
    assert list(loaded.task_input.alternative.responses.units) == list(
        response.units[row]
    )
    assert list(loaded.task_input.alternative.responses.genes) == list(
        response.genes[col]
    )
    assert (
        loaded.task_input.alternative.identity_inputs
        == np.asarray(alt.identity_inputs)[row].tolist()
    )
    assert {
        x.params["measure"]: x.value
        for x in loaded.score()
        if x.params["aggregation"] == "equal_donor"
    }["alternative_mse"] == 1.0


def test_no_overwrites_including_symlink(bundle_case, tmp_path):
    path, digest, matched, inputs = bundle_case
    alias = tmp_path / "alias.npz"
    alias.symlink_to(path)
    for destination in (path, alias):
        with pytest.raises(FileExistsError):
            save_prediction_bundle(destination, matched, inputs)
        with pytest.raises(FileExistsError):
            main([str(path), "--output", str(destination)])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "change",
    [
        "bad_hash",
        "bad_version",
        "extra_field",
        "string_training_list",
        "wrong_scale",
        "heldout_fit",
        "generic_nonconstant",
        "wrong_identity",
        "wrong_gene",
    ],
)
def test_corrupt_or_semantically_invalid_inputs_rejected(bundle_case, change):
    path, digest, _, _ = bundle_case
    if change == "bad_hash":
        with pytest.raises(ValueError, match="SHA-256"):
            load_prediction_bundle(path, expected_sha256="0" * 64)
        return
    array_updates = {}
    manifest_update = None
    if change == "bad_version":

        def manifest_update(m):
            return m.update(schema="unknown")

    if change == "extra_field":

        def manifest_update(m):
            return m["metadata"].update(unknown="x")

    if change == "string_training_list":

        def manifest_update(m):
            return m["metadata"]["fitted_donors"]["matched"].update(d1="d2")

    if change == "wrong_scale":

        def manifest_update(m):
            return m["scales"].update(generic="different")

    if change == "heldout_fit":

        def manifest_update(m):
            return m["metadata"]["fitted_donors"]["generic"].update(d1=["d1"])

    if change == "generic_nonconstant":
        array_updates["generic_values"] = np.zeros((4, 2))
        array_updates["generic_values"][1, 0] = 1
    if change == "wrong_identity":
        array_updates["matched_identity_inputs"] = np.array(["x", "x", "x", "x"])
    if change == "wrong_gene":
        array_updates["generic_genes"] = np.array(["g1", "unknown"])
    rewrite(path, array_updates=array_updates, manifest_update=manifest_update)
    with pytest.raises(ValueError):
        load_prediction_bundle(path).score()


@pytest.mark.parametrize(
    "value",
    [
        np.full((4, 2), np.nan),
        np.ones((4, 2), dtype=object),
        np.ones((4, 2), dtype=complex),
        np.ones((4, 2), dtype=bool),
        np.zeros((2, 2)),
    ],
)
def test_bad_numeric_arrays_rejected_even_when_rehashed(bundle_case, value):
    path, _, _, _ = bundle_case
    rewrite(path, array_updates={"matched_values": value})
    with pytest.raises(ValueError):
        load_prediction_bundle(path).score()


def test_untrusted_array_rewrite_rejected(bundle_case):
    path, _, _, _ = bundle_case
    rewrite(path, array_updates={"matched_values": np.zeros((4, 2))}, refresh=False)
    with pytest.raises(ValueError, match="SHA-256"):
        load_prediction_bundle(path)


@pytest.mark.parametrize("extra", ["truth_values.npy", "../outside.npy", "unknown.npy"])
def test_zip_member_whitelist(bundle_case, extra):
    path, _, _, _ = bundle_case
    rewrite(path, extra=extra)
    with pytest.raises(ValueError, match="members"):
        load_prediction_bundle(path)


@pytest.mark.parametrize("limit", [0, -1, True, 10])
def test_byte_limits(bundle_case, limit):
    path, _, matched, inputs = bundle_case
    with pytest.raises(ValueError):
        load_prediction_bundle(path, max_bytes=limit)
    with pytest.raises(ValueError):
        save_prediction_bundle(
            path.parent / "never.npz", matched, inputs, max_bytes=limit
        )
    assert not (path.parent / "never.npz").exists()


def test_nul_labels_do_not_silently_change(case, tmp_path):
    matched, params = case
    genes = pd.Index(["g1\x00", "g2"])
    params["truth"] = replace(params["truth"], genes=genes)
    params["matched_genes"] = genes
    params["alternative"] = replace(
        params["alternative"],
        responses=replace(params["alternative"].responses, genes=genes),
    )
    params["generic"] = replace(params["generic"], genes=genes)
    inputs = TASK_REGISTRY.validate_task_inputs(
        "perturbation_identity_response", params
    )
    with pytest.raises(ValueError, match="NUL"):
        save_prediction_bundle(tmp_path / "invalid.npz", matched, inputs)


def test_zero_width_unicode_rejected_before_allocation():
    from czbenchmarks.tasks.single_cell.perturbation_identity_bundle import _read_npy

    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<U0", "fortran_order": False, "shape": (100000,)}
    )
    with pytest.raises(ValueError):
        _read_npy(buffer.getvalue(), "truth_genes", 1024)


def test_numpy_string_identities_are_supported(bundle_case, tmp_path):
    _, _, matched, inputs = bundle_case
    inputs.alternative = replace(
        inputs.alternative,
        identity_inputs=np.asarray(inputs.alternative.identity_inputs, dtype="U"),
    )
    path = tmp_path / "numpy-identities.npz"
    save_prediction_bundle(path, matched, inputs)
    assert len(load_prediction_bundle(path).score()) == 15
