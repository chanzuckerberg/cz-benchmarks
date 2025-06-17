#!/usr/bin/env python3
"""
predict_fn_test.py  –  local smoke-test for an MLflow 3.1 PyFunc model.

Usage examples
--------------
# payload in a file
python predict_fn_test.py \
    --model-uri models:/single-cell-model/1 \
    --json-payload-file payload.json

# payload from STDIN
cat payload.json | python predict_fn_test.py \
    --model-uri runs:/abcdef123/model \
    --json-payload-file -
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import get_args

import mlflow.pyfunc
from mlflow.models.utils import PyFuncOutput  # union of legal output types
from mlflow.pyfunc.scoring_server import _parse_json_data  # MLflow 3.1 parser

LOGGER = logging.getLogger("test_predict")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------------------------------- #
# 1. Deserialize JSON payload
# --------------------------------------------------------------------------- #
def _deserialize_json(model, payload_path):
    """Deserialize the json payload to PyFuncInput types."""
    with sys.stdin if payload_path == "-" else open(Path(payload_path), "r") as fp:
        body = json.load(fp)
        LOGGER.info(f"Deserializing json payload: {body}")
        parsed = _parse_json_data(
            body,
            model.metadata,
            model.metadata.get_input_schema(),
        )

        return parsed


# --------------------------------------------------------------------------- #
# 2. Prediction
# --------------------------------------------------------------------------- #
def predict(model_uri: str, payload_path: str):
    """Load model, parse payload, run predict(), return result."""
    LOGGER.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)  # generic loader

    # Deserialize the json payload into the PyFuncInput types
    # accepted by PyFuncModel.predict()
    parsed = _deserialize_json(model, payload_path)

    LOGGER.info("Invoking predict(); data type=%s", type(parsed.data))
    return model.predict(parsed.data, params=parsed.params)  # pyfunc contract


# --------------------------------------------------------------------------- #
# 3. Output validation helper
# --------------------------------------------------------------------------- #
def _assert_valid_pyfunc_output(obj):
    """Ensure obj is one of the allowed PyFuncOutput types and non-empty."""
    allowed = get_args(PyFuncOutput)  # tuple of classes
    assert isinstance(obj, allowed), (
        f"predict() returned unsupported type {type(obj)}; must be one of {allowed}"
    )

    # Non-emptiness heuristic
    is_empty = False
    if obj is None:
        is_empty = True
    elif hasattr(obj, "__len__"):
        is_empty = len(obj) == 0  # works for list / DataFrame / ndarray
    elif hasattr(obj, "size"):  # numpy ndarray
        is_empty = obj.size == 0
    elif isinstance(obj, dict):
        is_empty = len(obj) == 0
    # For scalars / str we treat them as non-empty by default

    assert not is_empty, "predict() returned an empty result"

    LOGGER.info("Output validated: %s with non-empty content", type(obj).__name__)


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local MLflow predictor (JSON payload)"
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI (runs:/, models:/, local dir, …)",
    )
    parser.add_argument(
        "--json-payload-file",
        required=True,
        help="Path to JSON payload or '-' for stdin",
    )
    args = parser.parse_args()

    prediction = predict(args.model_uri, args.json_payload_file)
    _assert_valid_pyfunc_output(prediction)

    LOGGER.info("Prediction succeeded and passed validation")
