"""
TranscriptformerMLflowModel
==================================
Generic **file-URI → NumPy tensor** adapter for the MLflow PyFunc flavor.

This stub wraps any model that reads from a file URI and produces a NumPy tensor.
It relies on MLflow’s ModelSignature (from `model_spec.py`) to enforce that the
input DataFrame has the correct columns and types.

**USER MUST IMPLEMENT**:
   - `load_context(self, context)`
   - `_get_input(self, model_input: pd.DataFrame, params: Dict[str, Any])`
   - `_forward(self, input_obj: Any, params: Dict[str, Any])`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import tempfile

import anndata as ad

import mlflow.pyfunc
import numpy as np
import pandas as pd


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    """
    PyFunc wrapper: one-row DataFrame → NumPy tensor.

    Relies on MLflow’s signature enforcement for schema correctness.
    Only checks that exactly one row is passed and that the file exists.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Called **only once** when the model is loaded.

        Use `context.artifacts` (dict[name → local_path]) to load files,
        and `context.model_config` (dict) for any scalar metadata.

        Notes
        -----
        Typical implementation::

            ckpt   = context.artifacts["checkpoint"]
            vocab  = context.artifacts.get("vocab")
            variant = context.model_config.get("model_variant", "base")

            self.model = torch.load(ckpt, map_location="cpu")
            self.vocab = json.load(open(vocab)) if vocab else None
            self.variant = variant
        """
        # USER IMPLEMENTATION REQUIRED
        self.ckpt_path = Path(context.artifacts["checkpoint"]).resolve()
        self.model_variant = context.model_config.get("model_variant", "tf_sapiens")

    def predict(
        self,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Inference entry point (called per request).

        Parameters
        ----------
        model_input : pd.DataFrame
            A one-row DataFrame. Column names and dtypes are validated by MLflow.
        params : dict[str, Any], optional
            Runtime parameters (params schema and default values specified in ModelSignature).

        Returns
        -------
        np.ndarray
            Output tensor matching the saved ModelSignature.
        """
        params = params or {}

        # 1) Ensure exactly one row
        self._validate_input_df(model_input)

        # 2) Convert to framework-specific object
        input_obj = self._get_input(model_input, params)

        # 3) Run forward pass
        return self._forward(input_obj, params)

    def _get_input(
        self,
        model_input: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Any:
        """
        Convert the DataFrame + params into the object `_forward` expects.

        Parameters
        ----------
        model_input : pd.DataFrame
            One-row DataFrame (use `model_input.at[0, "input_uri"]` to get URI).
        params : dict[str, Any], optional
            Runtime parameters.

        Returns
        -------
        Any
            Framework-specific input (e.g., AnnData, PIL Image, ndarray).
        """
        # USER IMPLEMENTATION REQUIRED
        uri = model_input.at[0, "input_uri"]
        return uri

    def _forward(
        self,
        input_obj: Any,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """
        Execute the model’s forward pass.

        Parameters
        ----------
        input_obj : Any
            Object produced by `_get_input`, ready for inference.
        params : dict[str, Any], optional
            Runtime parameters.

        Returns
        -------
        np.ndarray
            NumPy array matching the ModelSignature’s output spec.
        """
        # USER IMPLEMENTATION REQUIRED
        input_uri = input_obj
        with tempfile.TemporaryDirectory() as td:
            out_h5ad = Path(td) / "embeddings.h5ad"
            self._run_cli(input_uri, out_h5ad, params)
            adata_out = ad.read_h5ad(out_h5ad)
            return np.asarray(adata_out.obsm["embeddings"], dtype=np.float32)

    def _default_batch_size(self) -> int:
        """Heuristic batch size per model family."""
        return {"tf_sapiens": 32, "tf_exemplar": 8, "tf_metazoa": 2}.get(
            self.model_variant, 8
        )

    def _run_cli(
        self,
        in_file: Path,
        out_file: Path,
        params: dict[str, Any],
    ) -> None:
        """Run **TranscriptFormer** inference for one file."""
        batch_size = params.get("batch_size", -1)
        if batch_size == -1:
            batch_size = self._default_batch_size()

        cmd = [
            "transcriptformer",
            "inference",
            "--checkpoint-path",
            str(self.ckpt_path),
            "--data-file",
            str(in_file),
            "--output-path",
            str(out_file.parent),
            "--output-filename",
            out_file.name,
            "--batch-size",
            str(batch_size),
            "--precision",
            str(params.get("precision", "16-mixed")),
            "--gene-col-name",
            str(params.get("gene_col_name", "ensembl_id")),
        ]
        pte = params.get("pretrained_embedding", "")
        if pte:
            cmd.extend(["--pretrained-embedding", str(pte)])

        subprocess.run(cmd, check=True)

    @staticmethod
    def _validate_input_df(df: pd.DataFrame) -> None:
        """
        Ensure the input DataFrame has exactly one row and that `input_uri`
        is a string path/URI that exists on disk.

        Raises
        ------
        ValueError
            If `df` does not contain exactly one row.
        TypeError
            If `input_uri` is not a string.
        FileNotFoundError
            If the file at `input_uri` does not exist.
        """
        # Row count check
        if len(df) != 1:
            raise ValueError("Model input must have exactly one row.")

        # Type & existence check for input_uri
        uri = df.at[0, "input_uri"]
        if not isinstance(uri, str):
            raise TypeError("input_uri must be a string path or URI.")
        if not Path(uri).expanduser().exists():
            raise FileNotFoundError(f"input_uri '{uri}' not found.")
