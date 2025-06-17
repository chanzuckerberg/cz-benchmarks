"""
TranscriptFormer → MLflow adapter (multi-file, column signature)
================================================================

* **Input**  : ``pd.DataFrame`` with one column **"input_uri"** and
               one row per H5AD file.
* **Output** : ``list[np.ndarray]`` – each element is an embedding
               matrix of shape ``(n_cells, 2048)`` (float32).
* **Params** : Runtime knobs validated by the `ParamSchema`.

The adapter:
1. Extracts every URI in the ``input_uri`` column.
2. Validates each file.
3. Invokes ``transcriptformer inference`` for each file.
4. Returns a list of embedding matrices preserving input order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import subprocess
import tempfile

import anndata as ad
import mlflow
import numpy as np
import pandas as pd


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    """
    MLflow *pyfunc* wrapper around the ``transcriptformer`` CLI.

    Parameters
    ----------
    model_variant : str
        Family identifier (e.g. ``"tf_sapiens"``); selects default batch
        size heuristics.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def __init__(self, model_variant: str) -> None:
        self.model_variant = model_variant

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Resolve the checkpoint directory once per worker."""
        self.ckpt_path = Path(context.artifacts["checkpoint"]).resolve()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_paths(df: pd.DataFrame) -> List[Path]:
        """
        Validate **shape** and **column name**, then return list[Path].

        Raises
        ------
        ValueError – if the DataFrame shape or column schema is wrong.
        """
        if list(df.columns) != ["input_uri"]:
            raise ValueError(
                "Input DataFrame must contain a single column named "
                "'input_uri'; got columns {df.columns.tolist()}"
            )
        return [Path(p).resolve() for p in df["input_uri"].tolist()]

    def _default_batch_size(self) -> int:
        """Heuristic batch size per model family."""
        return {"tf_sapiens": 32, "tf_exemplar": 8, "tf_metazoa": 2}.get(
            self.model_variant, 16
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

    # ------------------------------------------------------------------ #
    # Public inference                                                   #
    # ------------------------------------------------------------------ #
    # No type hints → avoid MLflow’s “type hints override explicit
    # signature” warning (docs).
    def predict(self, context, model_input, params=None):
        """
        Execute embedding inference for *each* input file.

        Parameters
        ----------
        model_input : pd.DataFrame
            1-column DataFrame named **"input_uri"**; any number of rows.
        params : dict, optional
            Runtime knobs. Validated upstream by the `ParamSchema`.

        Returns
        -------
        list[np.ndarray]
            One `(n_cells, 2048)` matrix per input row, preserving order.
        """
        params = params or {}

        # --- extract & download paths (router may have localised URIs) ---
        paths = self._extract_paths(model_input)
        outputs: List[np.ndarray] = []

        for in_path in paths:
            if not in_path.is_file():
                raise FileNotFoundError(in_path)

            with tempfile.TemporaryDirectory() as td:
                out_h5ad = Path(td) / "embeddings.h5ad"
                self._run_cli(in_path, out_h5ad, params)

                adata_out = ad.read_h5ad(out_h5ad)
                outputs.append(
                    np.asarray(adata_out.obsm["embeddings"], dtype=np.float32)
                )

        return outputs
