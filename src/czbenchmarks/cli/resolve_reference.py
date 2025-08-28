from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Optional, Mapping, Sequence
from anndata import AnnData

ANNDATA_REF_PREFIX = "@"


class AnnDataReference(BaseModel):
    """
    DataRef represents a structured reference to a specific slot or field within an AnnData object.
    References are denoted by a leading '@' and can point to various AnnData attributes:
        - '@X'                → adata.X (main data matrix)
        - '@obs'              → adata.obs (entire observations DataFrame)
        - '@obs:cell_type'    → adata.obs["cell_type"] (specific column in obs)
        - '@obsm:X_pca'       → adata.obsm["X_pca"] (specific key in obsm)
        - '@layers:counts'    → adata.layers["counts"] (specific layer)
        - '@var:gene_symbols' → adata.var["gene_symbols"] (specific column in var)
        - '@varm:some_key'    → adata.varm["some_key"] (specific key in varm)
        - '@uns:some_key'     → adata.uns["some_key"] (specific key in uns)
    Attributes:
        space (str): The AnnData attribute to reference ('X', 'obs', 'obsm', 'var', 'varm', 'layers', 'uns').
        key (Optional[str]): The key or column name within the specified space, if applicable.
    Methods:
        parse(value: str) -> DataRef:
            Parse a string reference (e.g., '@obs:cell_type') into a DataRef instance.
        resolve(adata: AnnData) -> Any:
            Resolve the reference against a given AnnData object, returning the referenced data.
    Raises:
        ValueError: If the reference format is invalid or unsupported.
        KeyError: If the specified key does not exist in the referenced AnnData attribute.
    """

    space: str
    key: Optional[str] = None

    @staticmethod
    def parse(ref_string: str) -> "AnnDataReference":
        if not ref_string.startswith(ANNDATA_REF_PREFIX):
            raise ValueError(f"Reference must start with '{ANNDATA_REF_PREFIX}'")

        ref_body = ref_string[len(ANNDATA_REF_PREFIX) :]
        if ":" in ref_body:
            data_space, data_key = ref_body.split(":", 1)
            data_space = data_space.strip()
            data_key = data_key.strip()
        else:
            data_space, data_key = ref_body.strip(), None
        return AnnDataReference(space=data_space, key=data_key)

    def resolve(self, anndata: AnnData) -> Any:
        data_space = self.space
        data_key = self.key

        if data_space == "X":
            if data_key is not None:
                raise ValueError("Ref '@X' does not take a key")
            return anndata.X

        if data_space == "obs":
            return anndata.obs if data_key is None else anndata.obs[data_key]

        if data_space == "var":
            return anndata.var if data_key is None else anndata.var[data_key]

        if data_space in ("obsm", "varm", "layers", "uns"):
            if data_key is None:
                raise ValueError(f"Ref '@{data_space}' requires a key")
            data_store = getattr(anndata, data_space)
            if data_key not in data_store:
                raise KeyError(f"Key '{data_key}' not found in anndata.{data_space}")
            return data_store[data_key]

        raise ValueError(
            f"Unsupported ref space '{data_space}'. Allowed: X, obs, obsm, var, varm, layers, uns"
        )


def is_anndata_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ANNDATA_REF_PREFIX)


def resolve_value_recursively(value: Any, anndata: AnnData) -> Any:
    if is_anndata_reference(value):
        return AnnDataReference.parse(value).resolve(anndata)
    if isinstance(value, AnnDataReference):
        return value.resolve(anndata)
    if isinstance(value, Mapping):
        return {
            param_name: resolve_value_recursively(param_value, anndata)
            for param_name, param_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [resolve_value_recursively(item, anndata) for item in value]
    return value


def resolve_task_parameters(
    task_params: dict[str, Any], anndata: AnnData
) -> dict[str, Any]:
    return resolve_value_recursively(task_params, anndata)
