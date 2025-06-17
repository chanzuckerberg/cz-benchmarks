#!/usr/bin/env python
"""
Generate a template JSON payload for an MLflow model without needing to spin up a server.

This script inspects a model's `ModelSignature` and emits one of two payload formats:

- **DataFrame (tabular) inputs** → `dataframe_split` JSON
- **Composite/array inputs**       → `"inputs"` JSON

Usage:
    python generate_mlflow_test_payload.py \
        --model-uri <MODEL_URI> \
        --json-payload-filepath test_input_payload.json

The generated `test_input_payload.json` can then be used directly with:

    cat test_input_payload.json | \
        mlflow models predict \
            --model-uri <MODEL_URI> \
            --content-type json \
            --input-path - \
            --output-path -

and will pass MLflow’s validation/serving stack without starting a REST server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlflow.artifacts import download_artifacts
from mlflow.models import Model
from mlflow.types.schema import ColSpec, Schema

# --------------------------------------------------------------------- #
# Placeholder for user-provided values                                       #
# --------------------------------------------------------------------- #
_PLACEHOLDER = "<FILL_ME>"


def _example_for_col(col: ColSpec) -> Any:
    """
    Generate a dummy value for a column based on its MLflow dtype.

    - Numeric types → 0
    - Boolean       → False
    - Array types   → [0]
    - Others (string, object, map, etc.) → <FILL_ME>
    """
    t = str(col.type)
    if t.startswith("array<"):
        return [0]
    if t in {"double", "integer"}:
        return 0
    if t == "boolean":
        return False
    return _PLACEHOLDER


def _payload_from_column_schema(schema: Schema) -> dict[str, Any]:
    """
    Build a JSON payload for tabular signatures (all ColSpec).
    Uses the "split" orientation (dataframe_split) with one row.
    """
    cols = list(schema)
    # Build a single row of placeholder values
    row = [_example_for_col(c) for c in cols]
    return {
        "dataframe_split": {
            "columns": [c.name or f"col_{i}" for i, c in enumerate(cols)],
            "data": [row],
        }
    }


def _payload_from_composite_schema(schema: Schema) -> dict[str, Any]:
    """
    Build a JSON payload for composite signatures (e.g. TensorSpec/array inputs).
    Emits a list under the top-level "inputs" key with generic placeholder values.
    """
    # For composite inputs we don't need type tokens—just a placeholder list
    return {"inputs": [_PLACEHOLDER]}


def build_payload(model_meta: Model) -> dict[str, Any]:
    """
    Inspect the ModelSignature on `model_meta` and dispatch to the
    appropriate payload builder.

    If no signature is present, defaults to a bare-bones:
        {"inputs": ["<FILL_ME>"]}

    Returns:
        A dict representing the JSON payload.
    """
    sig = model_meta.signature
    # If no signature, emit a generic placeholder payload
    if sig is None or sig.inputs is None:
        return {"inputs": [_PLACEHOLDER]}

    # Always a Schema object describing inputs
    input_schema = sig.inputs

    # Tabular (all ColSpec) → dataframe_split
    if all(isinstance(c, ColSpec) for c in input_schema):
        payload = _payload_from_column_schema(input_schema)
    # Composite/array/map inputs → top-level "inputs" key
    else:
        payload = _payload_from_composite_schema(input_schema)

    # Attach any params with defaults (or placeholders)
    if sig.params:
        payload["params"] = {
            p.name: (p.default if p.default is not None else _PLACEHOLDER)
            for p in sig.params
        }
    return payload


def main() -> None:
    """
    CLI entrypoint: parse args, load model metadata, generate payload, write to file.
    """
    parser = argparse.ArgumentParser(
        description="Generate example JSON payload for an MLflow model"
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Local path, runs:/, or models:/ URI of the MLflow model",
    )
    parser.add_argument(
        "--json-payload-filepath",
        default="test_input_payload.json",
        help="Output filepath for the generated JSON payload",
    )
    args = parser.parse_args()

    # Determine where to load the MLmodel metadata from
    if Path(args.model_uri).exists():
        mlmodel_path = Path(args.model_uri)
    else:
        # Download only metadata if remote (runs:/ or models:/)
        mlmodel_path = Path(download_artifacts(args.model_uri, dst_path=None))

    model_meta = Model.load(mlmodel_path)

    # Build and persist the payload JSON
    payload = build_payload(model_meta)
    out_path = Path(args.json_payload_filepath).resolve()
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ Wrote template JSON to {out_path}")


if __name__ == "__main__":
    main()
