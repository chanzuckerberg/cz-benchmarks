"""
Transcriptformer → MLflow pyfunc adapter.

* Exposes Transcriptformer inference through the MLflow Python Model API
* Expects a **pandas.DataFrame** with two columns:
      └─ input_file   : path to a .h5ad file to embed
      └─ output_file  : where to write the resulting embeddings
* Accepts an optional **params** dict (batch-wide) for runtime knobs.
* Returns a **pandas.Series** with the output paths (one-to-one with rows).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model that wraps the `transcriptformer inference` CLI.

    The class is loaded once per worker. Heavy artifacts (the checkpoint
    directory) are therefore fetched in `load_context()` rather than every
    prediction call.
    """

    def __init__(self, model_variant: str) -> None:
        self.model_variant = model_variant

    # --------------------------------------------------------------------- #
    #                       Lifecycle callbacks                              #
    # --------------------------------------------------------------------- #

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the checkpoint path from the MLflow artifacts.

        This method is called exactly once per worker when
        `mlflow.pyfunc.load_model()` instantiates the PythonModel.
        The path is cached on the instance so that `predict()` can
        reference it without repeatedly resolving artifacts.
        """
        self.checkpoint_path: Path = Path(context.artifacts["checkpoint_path"])

    # --------------------------------------------------------------------- #
    #                              Helpers                                   #
    # --------------------------------------------------------------------- #

    def _get_default_batch_size(self) -> int:
        """Return a GPU-specific default batch size for convenience."""
        return {"tf_sapiens": 32, "tf_exemplar": 8, "tf_metazoa": 2}.get(
            self.model_variant, 16
        )

    # --------------------------------------------------------------------- #
    #                             Prediction                                 #
    # --------------------------------------------------------------------- #

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.Series:
        """
        Perform inference for each row in *model_input*.

        Parameters
        ----------
        model_input
            A DataFrame with **input_file** and **output_file** columns.
        params
            Batch-wide runtime options; see README.  If omitted, sensible
            defaults are chosen.

        Returns
        -------
        pd.Series
            Each element is the path written by Transcriptformer.
        """
        params = params or {}
        results: list[str] = []

        # Basic validation – MLflow has already enforced dtypes for us, but
        # we check existence of the input files here.
        missing_cols = {"input_file", "output_file"} - set(model_input.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for _, row in model_input.iterrows():
            in_path = Path(row["input_file"]).expanduser()
            out_path = Path(row["output_file"]).expanduser()

            if not in_path.is_file():
                raise ValueError(f"Input file does not exist: {in_path}")

            # Resolve per-batch parameters (falling back to defaults).
            batch_size = params.get("batch_size", self._get_default_batch_size())
            gene_col = params.get("gene_col_name", "ensembl_id")
            precision = params.get("precision", "16-mixed")
            embed = params.get("pretrained_embedding")

            # Build `transcriptformer inference` CLI.
            cmd = [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                str(self.checkpoint_path),
                "--data-file",
                str(in_path),
                "--output-path",
                str(out_path.parent),
                "--output-filename",
                out_path.name,
                "--batch-size",
                str(batch_size),
                "--gene-col-name",
                gene_col,
                "--precision",
                precision,
            ]
            if embed:
                cmd += ["--pretrained-embedding", str(embed)]

            # Execute the subprocess; allow MLflow to surface any failure.
            subprocess.run(cmd, check=True)
            results.append(str(out_path))

        return pd.Series(results, name="output_file")
