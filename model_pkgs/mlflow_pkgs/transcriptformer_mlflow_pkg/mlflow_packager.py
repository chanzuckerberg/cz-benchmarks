"""
MLflow packager for TranscriptFormer (multi-file)
=================================================

Signature overview
------------------
inputs  : ColSpec("string", **"input_uri"**)
          – 1 column, any number of rows.
          – Each cell is a URI that points to an *.h5ad* file.

outputs : TensorSpec(float32, **(-1, 2048)**)
          – Describes **one** embedding matrix.
          – The Python model returns *list[np.ndarray]*; MLflow treats
            the list as repeating this spec.

params  : ParamSchema
          * `gene_col_name`        (string, default "ensembl_id")
          * `precision`            (string, default "16-mixed")
          * `pretrained_embedding` (string, default "")
          * `batch_size`           (integer, default -1)

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
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import DataType
from model_code.transcriptformer_mlflow_model import (
    TranscriptformerMLflowModel,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-variant",
        choices=["tf_sapiens", "tf_exemplar", "tf_metazoa"],
        required=True,
    )
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default="mlflow_models")
    p.add_argument(
        "--requirements",
        type=Path,
        default="transcriptformer_requirements.txt",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    save_path = args.output_dir / f"transcriptformer_{args.model_variant}"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------- ModelSignature ------------------------ #
    # INPUT  : DataFrame column "input_uri" (one H5AD per row).
    input_schema = Schema([ColSpec(DataType.string, "input_uri")])

    # OUTPUT : Single embedding matrix spec (row-count unknown).
    output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 2048))])

    # PARAMS : Runtime knobs (validated by MLflow).
    params_schema = ParamSchema(
        [
            ParamSpec("gene_col_name", "string", default="ensembl_id"),
            ParamSpec("precision", "string", default="16-mixed"),
            ParamSpec("pretrained_embedding", "string", default=""),
            ParamSpec(
                "batch_size",
                "integer",
                default=-1,  # -1 ⇒ use family-specific heuristic
            ),
        ]
    )

    signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
        params=params_schema,
    )

    # ------------------------ Save model --------------------------- #
    mlflow.pyfunc.save_model(
        path=str(save_path),
        python_model=TranscriptformerMLflowModel(args.model_variant),
        artifacts={"checkpoint": str(args.checkpoint_path)},
        pip_requirements=str(args.requirements),
        code_paths=[str((Path(__file__).parent / "model_code").resolve())],
        signature=signature,
        metadata={"tags": {"model_variant": args.model_variant}},
    )
    print(f"✓ Model packaged → {save_path}")


if __name__ == "__main__":
    main()
