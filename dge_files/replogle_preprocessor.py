# FIXME MICHELLE remove before merge
from typing import List

import numpy as np
import pandas as pd
import anndata as ad


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _compute_counts_and_genes(adata: ad.AnnData) -> None:
    if hasattr(adata.X, "sum"):
        umi_count = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        umi_count = np.sum(adata.X, axis=1)
    adata.obs["UMI_count"] = umi_count

    if hasattr(adata.X, "todense") or hasattr(adata.X, "tocsc"):
        ngenes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    else:
        ngenes = np.count_nonzero(adata.X, axis=1)
    adata.obs["ngenes"] = ngenes

    adata.obs["UMI_count"] = pd.to_numeric(adata.obs["UMI_count"], errors="coerce")


def _compute_percent_ribo(adata: ad.AnnData, gene_name_key: str = "gene") -> None:
    if gene_name_key not in adata.var:
        return
    gene_names = adata.var[gene_name_key].astype(str)
    ribo_mask = gene_names.str.startswith(("RPS", "RPL"))
    if ribo_mask.any():
        ribo_X = adata[:, ribo_mask].X
        ribo_sum = np.asarray(ribo_X.sum(axis=1)).ravel()
        denom = np.maximum(adata.obs["UMI_count"].values.astype(float), 1.0)
        adata.obs["percent_ribo"] = ribo_sum / denom


def _deduplicate_var_gene_names(adata: ad.AnnData, gene_name_key: str = "gene") -> None:
    if gene_name_key in adata.var:
        adata.var[gene_name_key] = adata.var[gene_name_key].astype("object")
        for dup_label in ["TBCE", "HSPA14"]:
            mask = adata.var[gene_name_key] == dup_label
            if mask.any():
                adata.var.loc[mask, gene_name_key] = (
                    dup_label + "_" + adata.var.loc[mask, :].index.astype(str)
                )

        if not adata.var[gene_name_key].is_unique:
            counts: dict[str, int] = {}
            new_names: List[str] = []
            for name, idx in zip(adata.var[gene_name_key].tolist(), adata.var.index.tolist()):
                c = counts.get(name, 0)
                new = name if c == 0 else f"{name}_{idx}"
                counts[name] = c + 1
                new_names.append(new)
            adata.var[gene_name_key] = new_names


def _drop_duplicate_obs_rows(adata: ad.AnnData) -> None:
    dup_mask = adata.obs.index.duplicated(keep="first")
    if dup_mask.any():
        keep_mask = ~dup_mask
        adata._inplace_subset_obs(keep_mask)


def _reformat_columns(adata: ad.AnnData) -> ad.AnnData:
    if "gene_id" in adata.obs:
        adata.obs = adata.obs.rename(columns={"gene_id": "condition"})
    if "gene" in adata.obs:
        adata.obs = adata.obs.rename(columns={"gene": "condition_name"})
    if "gene_name" in adata.var:
        adata.var = adata.var.rename(columns={"gene_name": "gene"})

    if "condition" in adata.obs:
        adata.obs["condition"] = adata.obs["condition"].astype(str)

    adata.var["ensembl_id"] = adata.var.index.astype(str)
    return adata


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    adata = _reformat_columns(adata)
    _compute_counts_and_genes(adata)
    _compute_percent_ribo(adata, gene_name_key="gene")
    _deduplicate_var_gene_names(adata, gene_name_key="gene")
    _drop_duplicate_obs_rows(adata)

    for key in ["UMI_count", "ngenes"]:
        if key in adata.obs:
            adata.obs[key] = _to_numeric(adata.obs[key])

    return adata


