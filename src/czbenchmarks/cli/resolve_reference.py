"""AnnData reference resolution utilities.

This module provides utilities for resolving string references to AnnData object attributes.
References use the '@' prefix to denote AnnData slots and support both single and multi-dataset contexts.

Supported reference formats:
    - "@" or "@adata"      → adata (entire AnnData object)
    - "@X"                 → adata.X (main data matrix)
    - "@obs"               → adata.obs (entire observations DataFrame)
    - "@obs:cell_type"     → adata.obs["cell_type"] (specific column)
    - "@obsm:X_pca"        → adata.obsm["X_pca"] (specific key in obsm)
    - "@layers:counts"     → adata.layers["counts"] (specific layer)
    - "@var:gene_symbols"  → adata.var["gene_symbols"] (specific column in var)
    - "@varm:some_key"     → adata.varm["some_key"] (specific key in varm)
    - "@uns:some_key"      → adata.uns["some_key"] (specific key in uns)
    - "@var_index"         → adata.var.index
    - "@obs_index"         → adata.obs.index

Multi-dataset indexed references:
    - "@0:X"               → adata_list[0].X (first dataset's X matrix)
    - "@1:obs:cell_type"   → adata_list[1].obs["cell_type"]

The main entry point is `resolve_anndata_references()` which handles both single
AnnData objects and lists of AnnData objects.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Union

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)

ANNDATA_REF_PREFIX = "@"


def is_anndata_reference(value: Any) -> bool:
    """Checks if a value is a string that looks like an AnnData reference."""
    return isinstance(value, str) and value.startswith(ANNDATA_REF_PREFIX)


def resolve_anndata_references(
    input_value: Any, adata_context: Union[AnnData, List[AnnData]]
) -> Any:
    """Recursively resolves AnnData references within nested data structures.

    Handles both single and multi-dataset contexts with clear error messages.

    Args:
        input_value: Value to resolve (string, list, dict, or primitive)
        adata_context: Single AnnData or list of AnnData objects

    Returns:
        Input value with all AnnData references resolved to actual data

    Raises:
        ValueError: If indexed reference used in single-dataset context
        IndexError: If dataset index is out of bounds
        KeyError: If key not found in specified AnnData attribute
    """
    is_multi_context = isinstance(adata_context, list)
    logger.debug(
        f"Resolving references: multi={is_multi_context}, "
        f"input_type={type(input_value).__name__}, value={input_value if not isinstance(input_value, (dict, list)) else '...'}"
    )

    def _resolve_single_reference(ref_string: str) -> Any:
        """Parses and resolves a single AnnData reference string."""
        logger.debug(f"Resolving single reference: {ref_string}")
        if not is_anndata_reference(ref_string):
            return ref_string

        body = ref_string[len(ANNDATA_REF_PREFIX) :]
        parts = body.split(":", 1)

        if parts[0].isdigit():
            if not is_multi_context:
                raise ValueError(
                    f"Indexed reference '{ref_string}' is not valid in a single-dataset context."
                )

            index = int(parts[0])
            if index >= len(adata_context):
                raise IndexError(
                    f"Dataset index {index} in reference '{ref_string}' is out of range for {len(adata_context)} datasets."
                )

            target_adata = adata_context[index]

            ref_string = ANNDATA_REF_PREFIX + (parts[1] if len(parts) > 1 else "")
        else:
            target_adata = adata_context[0] if is_multi_context else adata_context

        return _resolve_standard_reference(ref_string, target_adata)

    def _resolve_standard_reference(ref_string: str, adata: AnnData) -> Any:
        """Resolves a non-indexed reference against a single AnnData object."""
        body = ref_string[len(ANNDATA_REF_PREFIX) :]
        if body == "":
            logger.debug("Returning entire AnnData object")
            return adata

        parts = body.split(":", 1)
        object_name = parts[0]
        key = parts[1] if len(parts) > 1 else None
        logger.debug(
            f"Resolving standard reference: object_name={object_name}, key={key}"
        )

        if object_name == "var_index":
            if key:
                raise ValueError("Reference '@var_index' does not accept a key.")
            return adata.var.index
        if object_name == "obs_index":
            if key:
                raise ValueError("Reference '@obs_index' does not accept a key.")
            return adata.obs.index

        if object_name == "X":
            if key:
                raise ValueError("Reference '@X' does not accept a key.")
            return adata.X

        data_store = getattr(adata, object_name)

        if key is None:
            if object_name not in {"obs", "var"}:
                raise ValueError(f"Reference '@{object_name}' requires a key.")
            return data_store

        if key not in data_store:
            if object_name == "var" and key == "index":
                return adata.var.index
            if object_name == "obs" and key == "index":
                return adata.obs.index
            raise KeyError(f"Key '{key}' not found in adata.{object_name}.")

        return data_store[key]

    if is_anndata_reference(input_value):
        return _resolve_single_reference(input_value)

    if isinstance(input_value, Mapping):
        out = {}
        for k, v in input_value.items():
            rv = resolve_anndata_references(v, adata_context)
            if k in ("gene_index", "cell_index"):
                if isinstance(rv, pd.Series):
                    rv = pd.Index(rv)
                elif isinstance(rv, (list, np.ndarray)):
                    rv = pd.Index(rv)
            out[k] = rv
        return out

    if isinstance(input_value, list):
        return [resolve_anndata_references(item, adata_context) for item in input_value]

    if isinstance(input_value, tuple):
        return tuple(
            resolve_anndata_references(item, adata_context) for item in input_value
        )

    return input_value
