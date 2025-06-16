"""
Packager for TranscriptFormer (file-based API)
=============================================

* Creates a minimal `input_example` / `output_example`
  so `infer_signature()` captures dtypes without hand-coding.
* Adds a ParamSchema for batch-wide runtime knobs.
* Persists the model as an MLflow pyfunc artifact that expects
  **string URI(s)** to `.h5ad` files and returns **list[np.ndarray]`
  with shape `(n_cells, 2048)` per file.

Example
-------
python mlflow_packager.py \
    --model-variant tf_sapiens \
    --checkpoint-path /checkpoints/tf_sapiens \
    --output-dir mlflow_models
    --requirements transcriptformer_requirements.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec, ParamSchema, ParamSpec
from mlflow.types.schema import DataType
from model_code.transcriptformer_mlflow_model import TranscriptformerMLflowModel


# --------------------------------------------------------------------- #
# CLI utilities                                                         #
# --------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-variant",
        choices=["tf_sapiens", "tf_exemplar", "tf_metazoa"],
        required=True,
    )
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default="mlflow_models")
    p.add_argument(
        "--requirements", type=Path, default="transcriptformer-requirements.txt"
    )
    return p.parse_args()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()
    save_path = args.output_dir / f"transcriptformer_{args.model_variant}"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------- Explicit ModelSignature ---------------------- #
    # Input: list of strings (each a URI → local .h5ad)
    input_schema = Schema([TensorSpec(DataType.string.to_numpy(), shape=(-1,))])
    # Output: list of 2048-dim float32 embeddings
    output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 2048))])
    # Params: gene column, precision, pretrained_embedding, batch_size
    params_schema = ParamSchema(
        [
            ParamSpec("gene_col_name", "string", default="ensembl_id"),
            ParamSpec("precision", "string", default="16-mixed"),
            ParamSpec("pretrained_embedding", "string", default=""),
            ParamSpec("batch_size", "integer", default=16),
        ]
    )

    signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
        params=params_schema,
    )

    # ---------------------- Save model ------------------------------- #
    mlflow.pyfunc.save_model(
        path=str(save_path),
        python_model=TranscriptformerMLflowModel(args.model_variant),
        artifacts={"checkpoint": str(args.checkpoint_path)},
        pip_requirements=str(args.requirements),
        code_paths=[str((Path(__file__).parent / "model_code").resolve())],
        signature=signature,
        metadata={"tags": {"model_variant": args.model_variant}},
    )
    print("✓ Model packaged →", save_path)


if __name__ == "__main__":
    main()
