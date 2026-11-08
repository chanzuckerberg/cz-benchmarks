import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from czbenchmarks.datasets.perturbation_identity_cells import (
    aggregate_anndata_responses,
    PerturbationIdentityCellsDataset,
)
from czbenchmarks.datasets.types import Organism


KW = dict(
    donor_key="donor",
    target_key="target",
    control_key="target",
    control_value="NTC",
    context_key="context",
    clone_key="clone",
    target_catalog=["A", "B"],
    scored_genes=["g1", "g2"],
    chunk_size=2,
)


def fixture():
    # c1 target has two cells; c2 one cell: equal clone, not cell weighting.
    x = np.array(
        [[2, 3, 5], [3, 2, 15], [1, 1, 8], [1, 4, 5], [6, 2, 2], [4, 2, 4]],
        dtype=np.int64,
    )
    obs = pd.DataFrame(
        dict(
            donor=["d1"] * 6,
            clone=["c1"] * 3 + ["c2"] * 3,
            target=["NTC", "A", "A", "NTC", "A", "other"],
            context=["ctx"] * 6,
        ),
        index=[f"cell{i}" for i in range(6)],
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))


def run(a, **kwargs):
    return aggregate_anndata_responses(a, **(KW | kwargs))


def test_pooling_full_denominator_clone_weights_and_coverage():
    p = run(fixture())
    expected = (
        (
            np.log1p(np.array([4, 3]) / 30 * 10000)
            - np.log1p(np.array([2, 3]) / 10 * 10000)
        )
        + (
            np.log1p(np.array([6, 2]) / 10 * 10000)
            - np.log1p(np.array([1, 4]) / 10 * 10000)
        )
    ) / 2
    np.testing.assert_allclose(p.responses[0], expected)
    assert p.available.tolist() == [True, False]
    assert np.isnan(p.responses[1]).all()
    assert p.min_clone_cell_counts.tolist() == [1, 0]
    assert p.total_cell_counts.tolist() == [3, 0]
    assert p.clone_coverage.tolist() == [2, 0]
    assert p.manifest["excluded_noncatalog_targets"] == {"other": 1}
    assert p.manifest["input_total_umis"] == 70
    assert p.to_labeled_responses(p.available).values.shape == (1, 2)
    with pytest.raises(ValueError):
        p.to_labeled_responses(np.array([True, True]))


@pytest.mark.parametrize("fmt", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_dense_sparse_backed_and_loader_parity(tmp_path, fmt):
    a = fixture()
    expected = run(a)
    a.X = fmt(a.X)
    before = a.copy()
    result = run(a)
    np.testing.assert_allclose(result.responses, expected.responses, equal_nan=True)
    np.testing.assert_array_equal(
        a.X.toarray() if sparse.issparse(a.X) else a.X,
        before.X.toarray() if sparse.issparse(before.X) else before.X,
    )
    pd.testing.assert_frame_equal(a.obs, before.obs)
    pd.testing.assert_frame_equal(a.var, before.var)
    path = tmp_path / "cells.h5ad"
    a.write_h5ad(path)
    if sparse.isspmatrix_csc(a.X):
        backed = ad.read_h5ad(path, backed="r")
        try:
            with pytest.raises(ValueError, match="backed CSC"):
                run(backed)
        finally:
            backed.file.close()
        ds = PerturbationIdentityCellsDataset(path, Organism.HUMAN, **KW)
        with pytest.raises(ValueError, match="backed CSC"):
            ds.load_data()
        return
    backed = ad.read_h5ad(path, backed="r")
    try:
        np.testing.assert_allclose(
            run(backed).responses, expected.responses, equal_nan=True
        )
        assert backed.file.is_open
    finally:
        backed.file.close()
    ds = PerturbationIdentityCellsDataset(
        path, Organism.HUMAN, task_inputs_dir=tmp_path / "panel", **KW
    )
    ds.load_data()
    ds.validate()
    np.testing.assert_allclose(ds.panel.responses, expected.responses, equal_nan=True)
    out = ds.store_task_inputs()
    with np.load(out / "responses.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(
            arrays["responses"], expected.responses, equal_nan=True
        )
    assert json.loads((out / "labels.json").read_text())["genes"] == ["g1", "g2"]


@pytest.mark.parametrize("fmt", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_explicit_layers(tmp_path, fmt):
    a = fixture()
    a.layers["counts"] = fmt(a.X.copy())
    a.X = np.log1p(a.X)
    a.uns["log1p"] = {"base": None}
    with pytest.raises(ValueError, match="log1p"):
        run(a)
    expected = run(a, count_layer="counts")
    path = tmp_path / "layer.h5ad"
    a.write_h5ad(path)
    ds = PerturbationIdentityCellsDataset(
        path, Organism.HUMAN, **(KW | {"count_layer": "counts"})
    )
    if sparse.isspmatrix_csc(a.layers["counts"]):
        with pytest.raises(ValueError, match="backed CSC"):
            ds.load_data()
        return
    ds.load_data()
    np.testing.assert_allclose(ds.panel.responses, expected.responses, equal_nan=True)
    with pytest.raises(ValueError, match="count_layer"):
        run(a, count_layer="absent")


def test_permutations_and_expected_universe():
    a = fixture()
    p = run(a)
    q = run(
        a[[5, 2, 0, 4, 1, 3], [2, 0, 1]],
        expected_normalization_genes=["g2", "g3", "g1"],
    )
    np.testing.assert_allclose(p.responses, q.responses, equal_nan=True)
    with pytest.raises(ValueError, match="universe"):
        run(a, expected_normalization_genes=["g1", "g2"])
    with pytest.raises(ValueError, match="all scored_genes"):
        run(a, scored_genes=["missing"])


def test_missing_clone_target_is_unavailable_and_missing_controls_rejected():
    a = fixture()
    p = run(a[[0, 1, 2, 3, 5]])
    assert p.total_cell_counts[0] == 2
    assert p.min_clone_cell_counts[0] == 0
    assert p.clone_coverage[0] == 1
    assert not p.available[0]
    with pytest.raises(ValueError, match="missing matched controls"):
        run(a[[1, 2, 3, 4, 5]])


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 0.2, 2**53])
def test_bad_counts_even_on_excluded_cells(value):
    a = fixture()
    a.X = a.X.astype(float)
    a.X[-1, 0] = value
    with pytest.raises(ValueError):
        run(a)


def test_precision_and_total_overflow_and_zero_libraries():
    a = fixture()
    a.X = a.X.astype(np.float32)
    a.X[0, 0] = 2**24
    with pytest.raises(ValueError, match="precision"):
        run(a)
    a = fixture()
    a.X[:] = 2**52
    with pytest.raises(ValueError, match="aggregation"):
        run(a)
    a = fixture()
    a.X[-1] = 0
    with pytest.raises(ValueError, match="zero-library"):
        run(a)


@pytest.mark.parametrize(
    "change",
    [
        lambda a: setattr(a, "obs_names", ["dup"] * 6),
        lambda a: setattr(a, "var_names", ["dup"] * 3),
        lambda a: a.obs.__setitem__("donor", [None] + ["d1"] * 5),
        lambda a: a.obs.__setitem__("clone", [""] + ["c1"] * 5),
        lambda a: a.obs.__setitem__("context", ["other"] + ["ctx"] * 5),
    ],
)
def test_bad_metadata(change):
    a = fixture()
    change(a)
    with pytest.raises(ValueError):
        run(a)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target_catalog=["A", "A"]),
        dict(scored_genes=["g1", "g1"]),
        dict(target_catalog=["NTC", "A"]),
        dict(control_key="donor"),
        dict(clone_key="donor"),
        dict(chunk_size=0),
        dict(chunk_size=True),
    ],
)
def test_argument_collisions(kwargs):
    with pytest.raises(ValueError):
        run(fixture(), **kwargs)


def test_boolean_control_annotation_and_collision():
    a = fixture()
    a.obs["control"] = a.obs.target == "NTC"
    np.testing.assert_allclose(
        run(a, control_key="control", control_value=True).responses,
        run(a).responses,
        equal_nan=True,
    )
    a.obs.loc["cell1", "target"] = "NTC"
    with pytest.raises(ValueError, match="inconsistent"):
        run(a, control_key="control", control_value=True)


def test_custom_loader(tmp_path):
    from czbenchmarks.datasets.utils import load_custom_dataset

    path = tmp_path / "custom.h5ad"
    fixture().write_h5ad(path)
    ds = load_custom_dataset(
        "strict_cells",
        custom_dataset_kwargs={
            "_target_": "czbenchmarks.datasets.perturbation_identity_cells.PerturbationIdentityCellsDataset",
            "path": str(path),
            "organism": Organism.HUMAN,
            **KW,
        },
    )
    assert ds.panel.available.tolist() == [True, False]


def test_backed_csc_rejected_before_slice():
    from types import SimpleNamespace

    class UnreadableCSC:
        format = "csc"
        shape = (6, 3)

        def __getitem__(self, key):
            raise AssertionError("must reject before reading CSC values")

    a = fixture()
    proxy = SimpleNamespace(
        obs=a.obs,
        obs_names=a.obs_names,
        var_names=a.var_names,
        X=UnreadableCSC(),
        uns={},
        layers={},
    )
    with pytest.raises(ValueError, match="backed CSC"):
        run(proxy)


@pytest.mark.parametrize("layer", [None, "counts"])
def test_loader_csc_rejected_before_hdf5_count_reads(tmp_path, monkeypatch, layer):
    import h5py

    a = fixture()
    if layer is None:
        a.X = sparse.csc_matrix(a.X)
    else:
        a.layers[layer] = sparse.csc_matrix(a.X)
    path = tmp_path / "csc.h5ad"
    a.write_h5ad(path)
    original = h5py.Dataset.__getitem__
    reads = []

    def tracked(self, key):
        if self.name.endswith(("/data", "/indices", "/indptr")):
            reads.append(self.name)
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
    ds = PerturbationIdentityCellsDataset(
        path, Organism.HUMAN, **(KW | {"count_layer": layer})
    )
    with pytest.raises(ValueError, match="backed CSC"):
        ds.load_data()
    assert reads == []
    assert ds.panel is None
    # No live read-only handle is retained after the failure.
    with h5py.File(path, "r+"):
        pass


def test_sparse_duplicate_indices_are_safely_summed_without_mutation():
    a = fixture()
    dense = a.X.copy()
    # Represent each count as two duplicate stored half-counts (all integral).
    rows, cols = np.indices(a.shape)
    values = dense.ravel()
    indices = np.repeat(cols.ravel(), 2)
    data = np.column_stack([values // 2, values - values // 2]).ravel()
    a.X = sparse.csr_matrix((data, indices, np.arange(0, 37, 6)), shape=a.shape)
    assert not a.X.has_canonical_format
    original_data = a.X.data.copy()
    original_indices = a.X.indices.copy()
    np.testing.assert_allclose(
        run(a).responses, run(fixture()).responses, equal_nan=True
    )
    np.testing.assert_array_equal(a.X.data, original_data)
    np.testing.assert_array_equal(a.X.indices, original_indices)
    assert not a.X.has_canonical_format


def test_uint64_overflow_rejected_before_sparse_reduction():
    a = fixture()
    a.X = sparse.csr_matrix(a.X.astype(np.uint64))
    a.X.data[0] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError, match="precision"):
        run(a)


def test_control_only_donor_remains_in_full_catalog():
    a = fixture()
    a.obs.loc["cell3", "donor"] = "control_only"
    a.obs.loc["cell4", "clone"] = "c1"
    a.obs.loc["cell5", "clone"] = "c1"
    p = run(a)
    assert p.units.tolist() == [
        ("control_only", "A"),
        ("control_only", "B"),
        ("d1", "A"),
        ("d1", "B"),
    ]
    assert p.available.tolist() == [False, False, True, False]
    assert p.min_clone_cell_counts.tolist() == [0, 0, 3, 0]


def test_loader_reports_missing_public_anndata_io(tmp_path, monkeypatch):
    path = tmp_path / "cells.h5ad"
    fixture().write_h5ad(path)
    ds = PerturbationIdentityCellsDataset(path, Organism.HUMAN, **KW)
    monkeypatch.delattr(ad, "io")
    with pytest.raises(RuntimeError, match="AnnData >= 0.11"):
        ds.load_data()
