"""
Package a Transcriptformer checkpoint as an MLflow pyfunc model.

This script:
1. Builds a ModelSignature automatically from *real sample data*
   (infer_signature) to minimise drift
2. Adds a ParamSchema for batch-wide runtime knobs.
3. Saves the model with uv-compatible environment + code paths.

Example:
    python transcriptformer_mlflow_packager.py \
        --model-variant tf_sapiens \
        --checkpoint-path /checkpoints/tf_sapiens \
        --output-dir mlflow_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ParamSchema, ParamSpec
from model_code.transcriptformer_mlflow_model import TranscriptformerMLflowModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-variant", required=True, help="tf_sapiens | tf_exemplar | tf_metazoa"
    )
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default="mlflow_models")
    p.add_argument("--requirements", type=Path, default="requirements-mlflow-pkg.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_save_path = args.output_dir / f"transcriptformer_{args.model_variant}"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build a tiny input/output example to drive infer_signature()    #
    # ------------------------------------------------------------------ #
    input_example = pd.DataFrame(
        [
            {
                "input_file": "/tmp/sample_input.h5ad",
                "output_file": "/tmp/sample_output.h5ad",
            }
        ]
    )
    output_example = pd.Series(["/tmp/sample_output.h5ad"])

    # Infer columns + dtypes automatically to reduce maintenance.
    base_sig = infer_signature(model_input=input_example, model_output=output_example)

    # ------------------------------------------------------------------ #
    # 2. Create a ParamSchema for batch-wide options                     #
    # ------------------------------------------------------------------ #
    params_schema = ParamSchema(
        [
            ParamSpec("gene_col_name", "string", default="ensembl_id"),
            ParamSpec("precision", "string", default="16-mixed"),
            ParamSpec("pretrained_embedding", "string", default=""),
            ParamSpec("batch_size", "integer", default=16),
        ]
    )

    signature = ModelSignature(
        inputs=base_sig.inputs,
        outputs=base_sig.outputs,
        params=params_schema,
    )

    print(f"Packaging variant={args.model_variant} → {model_save_path}")

    mlflow.pyfunc.save_model(
        path=str(model_save_path),
        python_model=TranscriptformerMLflowModel(args.model_variant),
        artifacts={"checkpoint_path": str(args.checkpoint_path)},
        pip_requirements=str(args.requirements),
        code_paths=[str((Path(__file__).parent / "model_code").resolve())],
        signature=signature,
        input_example=input_example,  # stored as input_example.json
        metadata={"tags": {"model_variant": args.model_variant}},
    )
    print("✓ Model saved")


if __name__ == "__main__":
    main()
