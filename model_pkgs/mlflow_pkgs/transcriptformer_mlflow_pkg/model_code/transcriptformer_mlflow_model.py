"""
TranscriptFormer → MLflow adapter (file-based)
=============================================

* **Input**  : list[str] local paths to `.h5ad` files
               (the request handler wraps a single path in a list)
* **Output** : list[np.ndarray] – one embedding matrix per file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import tempfile

import anndata as ad
import mlflow
import numpy as np


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    """
    Wraps the ``transcriptformer inference`` CLI as an MLflow *pyfunc* model.

    Parameters
    ----------
    model_variant
        Identifier such as ``tf_sapiens`` – allows variant-specific defaults.
    """

    def __init__(self, model_variant: str) -> None:
        self.model_variant = model_variant

    # ------------------------------------------------------------------ #
    # MLflow lifecycle hook                                              #
    # ------------------------------------------------------------------ #

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Resolve the checkpoint directory once per worker."""
        self.ckpt_path: Path = Path(context.artifacts["checkpoint"]).resolve()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _default_batch_size(self) -> int:
        """Return a heuristic batch size based on model family."""
        return {"tf_sapiens": 32, "tf_exemplar": 8, "tf_metazoa": 2}.get(
            self.model_variant, 16
        )

    def _run_cli(self, in_file: Path, out_file: Path, params: dict[str, Any]) -> None:
        """Invoke TranscriptFormer CLI for one input file."""
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
            str(params.get("batch_size", self._default_batch_size())),
            "--precision",
            str(params.get("precision", "16-mixed")),
        ]
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------ #
    # Public inference                                                   #
    # ------------------------------------------------------------------ #

    # NOTE: Sometimes providing a type hint conflicts with the `ModelSignature`
    # because mlflow's support for generating a `ModelSignature` from type
    # hints is limited. This results in a strange situation where the type hint
    # generated `ModelSignature` overrides the explicit `ModelSignature`.
    # This can lead to unexpected behavior, where json payloads are insufficiently
    # validated by HTTP mlflow compatible serving platforms.
    #
    # RECOMMENDATION: Skip the type hints and specify a detailed `ModelSignature`
    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        """
        Parameters
        ----------
        model_input: List[str]
            List of local `.h5ad` paths prepared by the request handler.
        params: dict[str, Any]
            Batch-wide runtime knobs (`batch_size`, `precision`, …).

        Returns
        -------
        list[np.ndarray]
            One `(n_cells, 2048)` array per input file.
        """
        params = params or {}
        outputs: list[np.ndarray] = []

        for in_path_str in model_input:
            in_path = Path(in_path_str).resolve()
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
