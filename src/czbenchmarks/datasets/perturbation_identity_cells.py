"""Strict raw-cell input bridge for donor/target response panels, not a model task.

Count checks cannot establish raw-data provenance or that all gene-expression
features were supplied. The caller must establish both before using this API.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .dataset import Dataset

SCALE = "equal-clone mean of matched-control-subtracted log1p(pooled raw UMI CP10k; all supplied genes)"
_MAX_EXACT = 2**53


def _labels(values, name, unique=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identifiers, not a string")
    values = list(values)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty string identifiers")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} identifiers must be unique")
    return values


@dataclass(frozen=True)
class ResponsePanel:
    """Full donor × catalog coverage; unavailable responses are NaN, never zero.

    Counts and coverage have the same row order as ``units``. ``clone_counts``
    records each participating clone, including controls, for an auditable panel.
    No eligibility threshold or scoring policy is selected by this adapter.
    """

    units: pd.MultiIndex
    genes: pd.Index
    responses: np.ndarray
    available: np.ndarray
    min_clone_cell_counts: np.ndarray
    total_cell_counts: np.ndarray
    clone_coverage: np.ndarray
    clone_counts: pd.DataFrame
    manifest: dict
    scale: str = SCALE

    def to_labeled_responses(self, mask):
        """Select explicitly qualified rows for the existing response interface."""
        from ..tasks.single_cell.perturbation_identity_response import LabeledResponses

        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self.units),) or not mask.any():
            raise ValueError("mask must be a nonempty boolean row selection")
        if not self.available[mask].all():
            raise ValueError("selected rows contain unavailable responses")
        return LabeledResponses(
            self.responses[mask].copy(),
            self.units[mask].copy(),
            self.genes.copy(),
            self.scale,
        )

    def save(self, directory):
        """Write non-pickle numeric NPZ plus labels, clone counts and JSON manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "responses.npz",
            responses=self.responses,
            available=self.available,
            min_clone_cell_counts=self.min_clone_cell_counts,
            total_cell_counts=self.total_cell_counts,
            clone_coverage=self.clone_coverage,
        )
        (directory / "labels.json").write_text(
            json.dumps(
                {
                    "units": self.units.tolist(),
                    "genes": self.genes.tolist(),
                    "scale": self.scale,
                },
                indent=2,
            )
        )
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        self.clone_counts.to_csv(directory / "clone_counts.csv", index=False)
        return directory


def aggregate_anndata_responses(
    adata,
    *,
    donor_key,
    target_key,
    control_key,
    control_value,
    context_key,
    target_catalog,
    scored_genes,
    clone_key=None,
    count_layer=None,
    expected_normalization_genes=None,
    chunk_size=1024,
):
    """Pool raw UMIs, normalize across ALL features, subtract matched controls.

    One context is required. All donor/clone groups in the input participate,
    including groups represented only by controls or excluded targets. Controls
    are required for every group. Any missing clone-target makes the donor-target
    response unavailable. Clones are averaged equally, regardless of cell counts.
    ``control_key`` may equal ``target_key`` (e.g. control_value="NTC"); otherwise
    use an explicit metadata column. No donor IDs are inferred from barcodes.
    All cells, including excluded targets, undergo strict count validation.

    Memory scales with row chunk × input features and groups × scored features,
    not the complete cell expression matrix. Backed CSC is rejected because
    AnnData row slicing can load its complete sparse arrays; use backed CSR.
    Caller-owned AnnData is not changed
    or closed. A backed AnnData already loaded by its caller may have eager layers;
    use the Dataset loader for bounded H5AD layer access.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    catalog = _labels(target_catalog, "target_catalog", unique=True)
    genes = _labels(scored_genes, "scored_genes", unique=True)
    feature_names = _labels(adata.var_names, "var_names", unique=True)
    _labels(adata.obs_names, "obs_names", unique=True)
    keys = [donor_key, target_key, context_key] + (
        [] if clone_key is None else [clone_key]
    )
    _labels(keys, "metadata keys", unique=True)
    if not isinstance(control_key, str) or not control_key.strip():
        raise ValueError("control_key must be a nonempty string")
    if control_key in keys and control_key != target_key:
        raise ValueError("control_key collides with another metadata role")
    if not adata.obs.columns.is_unique:
        raise ValueError("obs metadata column identifiers must be unique")
    if not isinstance(control_value, (str, bool, np.bool_)) or (
        isinstance(control_value, str) and not control_value.strip()
    ):
        raise ValueError("control_value must be a nonempty string or boolean")
    for key in set(keys + [control_key]):
        if key not in adata.obs:
            raise ValueError(f"missing metadata column: {key}")
        if adata.obs[key].isna().any():
            raise ValueError(f"missing metadata values: {key}")
    donors = _labels(adata.obs[donor_key], donor_key)
    targets = _labels(adata.obs[target_key], target_key)
    contexts = _labels(adata.obs[context_key], context_key)
    clones = (
        _labels(adata.obs[clone_key], clone_key)
        if clone_key
        else ["__pooled__"] * len(donors)
    )
    if len(set(contexts)) != 1:
        raise ValueError("exactly one context is required")
    control_values = adata.obs[control_key].tolist()
    if isinstance(control_value, str):
        _labels(control_values, control_key)
    elif any(not isinstance(v, (bool, np.bool_)) for v in control_values):
        raise ValueError("boolean control_value requires boolean control metadata")
    controls = np.asarray([v == control_value for v in control_values])
    control_targets = {t for t, c in zip(targets, controls) if c}
    if control_targets.intersection(catalog):
        raise ValueError("control target labels collide with target_catalog")
    if any(t in control_targets and not c for t, c in zip(targets, controls)):
        raise ValueError("target label has inconsistent control annotations")
    feature_index = pd.Index(feature_names)
    positions = feature_index.get_indexer(genes)
    if (positions < 0).any():
        raise ValueError("all scored_genes must be present; no silent intersection")
    if expected_normalization_genes is not None:
        expected = _labels(
            expected_normalization_genes, "expected_normalization_genes", unique=True
        )
        if set(expected) != set(feature_names):
            raise ValueError("normalization gene universe differs from expected genes")
    if count_layer is None:
        if "log1p" in adata.uns:
            raise ValueError(
                "X carries log1p normalization metadata; use an explicit raw count layer"
            )
        matrix = adata.X
    else:
        if (
            not isinstance(count_layer, str)
            or not count_layer.strip()
            or count_layer not in adata.layers
        ):
            raise ValueError("count_layer must name an existing raw count layer")
        matrix = adata.layers[count_layer]
    if matrix is None or matrix.shape != (len(donors), len(feature_names)):
        raise ValueError("count matrix shape must match observation and feature labels")
    if not sparse.issparse(matrix) and getattr(matrix, "format", None) == "csc":
        raise ValueError(
            "backed CSC is not bounded-memory row-readable; convert to CSR H5AD "
            "with a separate memory-budgeted preprocessing step, or provide in-memory CSC"
        )
    donor_order = sorted(set(donors))
    groups = sorted(set(zip(donors, clones)))
    catalog_set = set(catalog)
    pools, cell_counts, excluded = {}, {}, {}
    total_umis = 0
    for start in range(0, len(donors), chunk_size):
        block = matrix[start : start + chunk_size, :]
        if sparse.issparse(block):
            values = block.data
        else:
            values = np.asanyarray(block)
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise ValueError("masked counts are not allowed")
        if (
            values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("counts must be finite nonnegative real numbers")
        if (values != np.floor(values)).any():
            raise ValueError("counts must be integral raw UMIs, not normalized values")
        precision_limit = _MAX_EXACT
        if values.dtype.kind == "f":
            precision_limit = min(
                precision_limit, 2 ** (np.finfo(values.dtype).nmant + 1)
            )
        if (values >= precision_limit).any():
            raise ValueError("counts exceed safe integer precision")
        # Bound every subsequent sum before sparse duplicate coalescing or casting.
        block_sum = np.sum(values, dtype=np.float64)
        if block_sum >= _MAX_EXACT or total_umis + int(block_sum) >= _MAX_EXACT:
            raise ValueError("total counts exceed safe exact integer aggregation")
        total_umis += int(block_sum)
        block = block.astype(np.float64, copy=True)
        if sparse.issparse(block):
            block = block.toarray()
        else:
            block = np.asarray(block)
        totals = block.sum(axis=1)
        if (totals == 0).any():
            raise ValueError("zero-library cells cannot be normalized")
        for offset, library_size in enumerate(totals):
            i = start + offset
            target = None if controls[i] else targets[i]
            if target is not None and target not in catalog_set:
                excluded[target] = excluded.get(target, 0) + 1
                continue
            key = (donors[i], clones[i], target)
            if key not in pools:
                pools[key] = np.zeros(len(genes) + 1)
                cell_counts[key] = 0
            pools[key][:-1] += block[offset, positions]
            pools[key][-1] += library_size
            cell_counts[key] += 1
    for donor, clone in groups:
        if (donor, clone, None) not in pools:
            raise ValueError(
                f"missing matched controls for donor={donor!r}, clone={clone!r}"
            )
    normalized = {
        key: np.log1p(pool[:-1] / pool[-1] * 10000) for key, pool in pools.items()
    }
    units = pd.MultiIndex.from_product(
        [donor_order, catalog], names=["donor", "target"]
    )
    responses = np.full((len(units), len(genes)), np.nan)
    minimum = np.zeros(len(units), dtype=np.int64)
    total_cells = np.zeros(len(units), dtype=np.int64)
    coverage = np.zeros(len(units), dtype=np.int64)
    clone_rows = []
    for donor, clone in groups:
        for target in [None] + catalog:
            key = (donor, clone, target)
            clone_rows.append(
                {
                    "donor": donor,
                    "clone": clone,
                    "target": target,
                    "is_control": target is None,
                    "cell_count": cell_counts.get(key, 0),
                    "total_umis": int(pools[key][-1]) if key in pools else 0,
                }
            )
    for row, (donor, target) in enumerate(units):
        donor_clones = [c for d, c in groups if d == donor]
        counts = [cell_counts.get((donor, clone, target), 0) for clone in donor_clones]
        minimum[row] = min(counts)
        total_cells[row] = sum(counts)
        coverage[row] = sum(n > 0 for n in counts)
        if minimum[row] > 0:
            responses[row] = np.mean(
                [
                    normalized[donor, clone, target] - normalized[donor, clone, None]
                    for clone in donor_clones
                ],
                axis=0,
            )
    manifest = {
        "schema_version": 1,
        "context": contexts[0],
        "scale": SCALE,
        "count_source": "X" if count_layer is None else f"layers/{count_layer}",
        "metadata_keys": {
            "donor": donor_key,
            "target": target_key,
            "control": control_key,
            "context": context_key,
            "clone": clone_key,
        },
        "control_value": bool(control_value)
        if isinstance(control_value, np.bool_)
        else control_value,
        "target_catalog": catalog,
        "scored_genes": genes,
        "normalization_genes": feature_names,
        "expected_normalization_genes_checked": expected_normalization_genes
        is not None,
        "input_cells": len(donors),
        "control_cells": int(controls.sum()),
        "included_target_cells": int(total_cells.sum()),
        "excluded_noncatalog_cells": sum(excluded.values()),
        "excluded_noncatalog_targets": dict(sorted(excluded.items())),
        "input_total_umis": total_umis,
        "clones_per_donor": {d: sum(g[0] == d for g in groups) for d in donor_order},
        "limitations": "Numeric validation cannot prove raw-data provenance or a complete gene-expression universe. Eligibility and scoring are caller policy.",
    }
    return ResponsePanel(
        units,
        pd.Index(genes, name="gene"),
        responses,
        minimum > 0,
        minimum,
        total_cells,
        coverage,
        pd.DataFrame(clone_rows),
        manifest,
    )


class PerturbationIdentityCellsDataset(Dataset):
    """Hydra/load_custom_dataset-compatible H5AD loader; no retained file handles.

    ``panel`` is populated by ``load_data``. Unlike SingleCellDataset this object
    does not retain a cell-level ``adata`` and is not an expression-prediction
    task input. H5AD expression data, including an explicit layer, is read in
    row chunks. Metadata and the resulting small aggregated panel are in memory.
    """

    def __init__(self, path, organism, task_inputs_dir=None, **aggregation_kwargs):
        """Configure a local H5AD input and its response-panel transformation.

        Args:
            path: Existing H5AD file containing raw counts and cell metadata.
            organism: A ``czbenchmarks.datasets.types.Organism`` enum value.
            task_inputs_dir: Output directory for the aggregated response panel.
            aggregation_kwargs: Arguments to ``aggregate_anndata_responses``.
        """
        super().__init__("perturbation_identity_cells", path, organism, task_inputs_dir)
        self.aggregation_kwargs = aggregation_kwargs
        self.panel = None

    def load_data(self):
        if not all(
            hasattr(getattr(ad, "io", None), name)
            for name in ("read_elem", "sparse_dataset")
        ):
            raise RuntimeError(
                "The bounded H5AD response loader requires AnnData >= 0.11 "
                "with public anndata.io.read_elem and sparse_dataset APIs. "
                "Use a compatible environment or the in-memory aggregation API."
            )
        # read_h5ad(backed='r') eagerly loads layers in some AnnData versions.
        # Open only metadata and the requested expression element instead.
        self.panel = None
        with h5py.File(self.path, "r") as handle:
            obs = ad.io.read_elem(handle["obs"])
            var = ad.io.read_elem(handle["var"])
            layer = self.aggregation_kwargs.get("count_layer")
            source = "X" if layer is None else f"layers/{layer}"
            if layer is not None and (
                not isinstance(layer, str) or not layer.strip() or "/" in layer
            ):
                raise ValueError("count_layer must name a direct H5AD layer")
            if source not in handle:
                raise ValueError(f"missing count matrix: {source}")
            element = handle[source]
            if (
                isinstance(element, h5py.Group)
                and element.attrs.get("encoding-type") == "csc_matrix"
            ):
                raise ValueError(
                    "backed CSC is not bounded-memory row-readable; convert to CSR H5AD "
                    "with a separate memory-budgeted preprocessing step"
                )
            matrix = (
                ad.io.sparse_dataset(element)
                if isinstance(element, h5py.Group)
                else element
            )
            uns = {"log1p": True} if "uns/log1p" in handle else {}
            proxy = SimpleNamespace(
                obs=obs,
                obs_names=obs.index,
                var_names=var.index,
                uns=uns,
                X=matrix if layer is None else None,
                layers={} if layer is None else {layer: matrix},
            )
            panel = aggregate_anndata_responses(proxy, **self.aggregation_kwargs)
        self.panel = panel

    def _validate(self):
        if self.panel is None:
            raise ValueError("load_data must successfully populate panel first")

    def store_task_inputs(self):
        self._validate()
        return self.panel.save(self.task_inputs_dir)
