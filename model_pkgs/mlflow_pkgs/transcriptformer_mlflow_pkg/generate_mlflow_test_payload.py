#!/usr/bin/env python
"""
Create a **template REST-payload** (`test_input_payload.json`) for any
MLflow-packaged model.  The file can then be fed to

    cat test_input_payload.json | \
        mlflow models predict  <MODEL_URI> \
            --content-type json --input-path - --output-path -

to verify that the payload round-trips through MLflow’s validation and
scoring stack without spinning up `mlflow models serve`.

The script works for:

* local directories produced by `mlflow.pyfunc.save_model`
* run URIs  (e.g.  `runs:/<run-id>/model`)
* model registry URIs  (e.g.  `models:/MyModel/5`)

Only **metadata is loaded** – no large weights – so the script is quick
and safe to run in CI.

------------------------------------------------------------------------
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
#                       Helper: simple dtype ➜ placeholder              #
# --------------------------------------------------------------------- #
_PLACEHOLDER = "<FILL_ME>"


def _example_for_col(col: ColSpec) -> Any:
    """Return a dummy value based on MLflow dtype."""
    t = str(col.type)
    if t.startswith("array<"):
        return [0]  # generic numeric list
    if t in {"double", "integer"}:
        return 0
    if t == "boolean":
        return False
    return _PLACEHOLDER  # string / object / map / csr_matrix …


# --------------------------------------------------------------------- #
#               Build JSON payload from a column-based schema           #
# --------------------------------------------------------------------- #
def _payload_from_column_schema(schema: Schema) -> dict[str, Any]:
    cols = list(schema.inputs)
    row = [_example_for_col(c) for c in cols]
    return {
        "dataframe_split": {
            "columns": [c.name or f"col_{i}" for i, c in enumerate(cols)],
            "data": [row],
        }
    }


# --------------------------------------------------------------------- #
#           Build JSON payload for composite / object schema            #
# --------------------------------------------------------------------- #
def _payload_from_composite_schema(schema: Schema) -> dict[str, Any]:
    # We emit the raw type token so users know what to replace.
    token = str(schema.inputs[0].type)
    return {"inputs": [f"{token}::{_PLACEHOLDER}"]}


# --------------------------------------------------------------------- #
#                              Core logic                               #
# --------------------------------------------------------------------- #
def build_payload(model_meta: Model) -> dict[str, Any]:
    sig = model_meta.signature
    if sig is None:
        return {"inputs": [_PLACEHOLDER]}  # no signature → generic stub

    # Determine input schema flavour (tabular vs composite)
    if isinstance(sig.inputs, Schema) and all(
        isinstance(c, ColSpec) for c in sig.inputs
    ):
        payload = _payload_from_column_schema(sig)
    else:
        payload = _payload_from_composite_schema(sig)

    # Add params (with defaults if present)
    if sig.params is not None:
        payload["params"] = {
            p.name: (p.default if p.default is not None else _PLACEHOLDER)
            for p in sig.params
        }
    return payload


# --------------------------------------------------------------------- #
#                        CLI entry-point                                #
# --------------------------------------------------------------------- #
def main() -> None:
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
        help="Filename to write (cwd). Default: %(default)s",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Resolve metadata (no big weights download)                      #
    # ------------------------------------------------------------------ #
    # For local path we can load directly; for remote / registry we download
    # metadata first (tiny).
    if Path(args.model_uri).exists():
        mlmodel_path = Path(args.model_uri)
    else:
        # Download only metadata; artifacts module handles Registry & runs: URIs
        mlmodel_path = Path(download_artifacts(args.model_uri, dst_path=None))

    model_meta = Model.load(mlmodel_path)

    # ------------------------------------------------------------------ #
    # 2. Build payload and write to file                                 #
    # ------------------------------------------------------------------ #
    payload = build_payload(model_meta)
    out_path = Path(args.json_payload_filepath).resolve()
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"✓ Wrote template JSON to {out_path}")


if __name__ == "__main__":
    main()
