#!/usr/bin/env python3
"""
mlflow_packager.py
==================

Package a *PyFunc* ``PythonModel`` **and** prove it can execute one complete
inference.

---------------------------------------------------------------------------
PROJECT CONVENTIONS
---------------------------------------------------------------------------
1. **Artifacts live in `model_data/`**

   * Every ``--artifact`` value is a *path **inside*** the `model_data/`
     folder **without** the prefix.  
     Examples::

         --artifact checkpoint=tf_sapiens/weights.ckpt
         --artifact vocab=vocabs/assay_vocab.json

   * If the user *does* include ``model_data/`` (legacy habits), the script
     strips it automatically, but best-practice is to omit it.
   * Absolute paths (``/tmp/weights.ckpt``) and remote URIs (``s3://…``) are
     forbidden—the packager will raise—so the same CLI works locally and in CI
     after CI downloads all blobs into `model_data/`.

2. **The input example is *real***  

   * `INPUT_EXAMPLE` in `model_spec.py` must feed directly into
     ``model.predict``.
   * The script stages the model to a **temporary directory**, reloads it,
     and runs this tiny forward-pass.  Packaging succeeds only if that call
     completes within the node’s resource limits.

3. **Static drift check stays**  

   * A millisecond-fast check validates column names, params, and common
     dtypes before any heavyweight library loads.
---------------------------------------------------------------------------
CLI
---------------------------------------------------------------------------
usage::
    python mlflow_packager.py
        --model-class MODULE:Class
        --artifact NAME=REL_PATH [--artifact NAME=REL_PATH ...]
        [--out-dir DIR]
        [--model-config-json JSON]
        [--model-tag KEY=VALUE ...]
        [--json-payload-filepath PATH]

CLI SWITCHES
~~~~~~~~~~~~
--model-class
    Fully-qualified ``PythonModel`` subclass  
    (e.g. ``model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel``).

--artifact
    **Repeatable.** Path *inside* `model_data/` copied into
    ``context.artifacts``.  Must exist; remote URIs are disallowed.

--model-config-json
    JSON string stored verbatim in ``context.model_config``.

--model-tag
    **Repeatable** ``KEY=VALUE`` pairs added as MLflow tags (if a run is
    active).

--out-dir
    Destination directory for the final, validated package.  It must not
    pre-exist; the script atomically moves the staged directory.

--json-payload-filepath
    Override the payload used for the runtime forward-pass **without**
    touching the embedded `INPUT_EXAMPLE`.  CI supplies a tiny 
    JSON file so tests stay fast; authors can ignore the flag.

---------------------------------------------------------------------------
EXAMPLES (TRANSCRIPTFORMER)
---------------------------------------------------------------------------
1) Minimal::
      python mlflow_packager.py \\
          --model-class model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel \\
          --artifact checkpoint=tf_sapiens \\
          --out-dir mlflow_model_artifact

2) Multiple artifacts + tags::
      python mlflow_packager.py \\
          --model-class model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel \\
          --artifact checkpoint=tf_sapiens \\
          --artifact vocab=vocabs/assay_vocab.json \\
          --model-config-json '{"model_variant":"tf_sapiens"}' \\
          --model-tag model_variant=tf_sapiens \\
          --model-tag stage=production \\
          --out-dir mlflow_model_artifact

3) Example for **CI using its own input payload**::
      python mlflow_packager.py \\
          --model-class model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel \\
          --artifact checkpoint=tf_sapiens \\
          ---json-payload-filepath transcriptformer/smoke-test-payload.json \\
          --out-dir mlflow_model_artifact
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import (
    convert_input_example_to_serving_input,
    validate_serving_input,
    validate_schema,
)

LOGGER = logging.getLogger("mlflow_packager")
_MODEL_DATA_ROOT = Path("model_data").resolve()


# --------------------------------------------------------------------------- #
# CLI helpers                                                                 #
# --------------------------------------------------------------------------- #
def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments (see module doc-string for semantics)."""
    parser = argparse.ArgumentParser(
        prog="mlflow_packager.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Package an MLflow model directory and execute a tiny "
        "forward-pass to guarantee runtime correctness.",
    )
    parser.add_argument("--model-class", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=REL_PATH",
        help="Repeatable.  File/dir under model_data/ copied into context.artifacts.",
    )
    parser.add_argument("--model-config-json", default="{}")
    parser.add_argument("--model-tag", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--json-payload-filepath",
        metavar="PATH",
        type=Path,
        help="Override INPUT_EXAMPLE for the forward-pass with a JSON file "
        "(e.g. edited input_example.json).  Static drift check still runs on "
        "the embedded example.",
    )
    parser.add_argument("--out-dir", default="mlflow_model_artifact")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #
def _validate_artifacts(artifacts: Dict[str, str]) -> None:
    """
    Ensure each artifact is a *relative* path that resolves to an existing file
    or directory **inside model_data/**.

    Disallowed:
      • Paths that start with "model_data/"      → instruct user to drop prefix
      • Absolute paths ("/tmp/ckpt.pt")
      • Remote URIs ("s3://bucket/key")
      • Parent escapes ("../other_folder/foo")
    """
    for name, rel_path in artifacts.items():
        p = Path(rel_path)

        # 1) prefix check ---------------------------------------------------
        if p.parts and p.parts[0] == "model_data":
            raise ValueError(
                f"--artifact {name}={rel_path!s} should be given WITHOUT the "
                "leading 'model_data/' prefix; use the path *inside* "
                "the folder, e.g. '--artifact checkpoint=tf_sapiens'"
            )

        # 2) absolute / remote / parent-escape checks -----------------------
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(
                f"--artifact {name}={rel_path!s} must be a path inside "
                "model_data/, not an absolute or parent path."
            )

        # 3) path must exist inside model_data/ ----------------------------
        abs_p = (_MODEL_DATA_ROOT / p).resolve()
        if not abs_p.exists():
            raise FileNotFoundError(
                f"--artifact {name}={rel_path!s} not found under model_data/"
            )

        # Store the absolute path so mlflow saves by copy, never 'downloads'
        artifacts[name] = str(abs_p)


def _static_drift_check(sig: ModelSignature, ex: Tuple[Any, Dict[str, Any]]) -> None:
    """
    Fast schema sanity-check that runs *before* heavyweight libs load.
    """
    data, params = ex

    # model input validation
    validate_schema(data=data, expected_schema=sig.inputs)

    # runtime params validation
    spec_params = [p.name for p in getattr(sig.params, "inputs", sig.params or [])]
    extra = params.keys() - spec_params
    if extra:
        raise ValueError(f"Param drift – unrecognized keys {sorted(extra)}")
    # missing keys are allowed: MLflow will fill defaults


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_cli(argv)

    # import target model
    module_path, class_name = args.model_class.split(":", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    # import signature & example
    import model_spec  # type: ignore

    signature: ModelSignature = model_spec.MODEL_SIGNATURE  # type: ignore
    embedded_input_example: Tuple[Any, Dict[str, Any]] = model_spec.INPUT_EXAMPLE  # type: ignore

    # validate artifacts
    artifacts = dict(kv.split("=", 1) for kv in args.artifact or [])
    if not artifacts:
        LOGGER.error("At least one --artifact NAME=REL_PATH is required.")
        sys.exit(1)
    _validate_artifacts(artifacts)

    # static drift check is done against the embedded example
    # because it is fast - validation before packaging starts.
    # The static drift checks only runs against the embedded example
    # because the embedded example is what shows up in the docs and
    # model registry UI - which are user facing - and that is what
    # needs to be consistent with the `ModelSignature`
    try:
        _static_drift_check(signature, embedded_input_example)
    except Exception:
        LOGGER.exception("Static drift check failed!")
        sys.exit(1)

    # load the json payload from the example or from a file path
    # supplied on the command line
    if args.json_payload_filepath:
        json_payload_filepath = args.json_payload_filepath.expanduser().resolve()
        LOGGER.info(
            f"Running packaging and inference with custom json payload at path: {json_payload_filepath}"
        )

        with json_payload_filepath.open() as f:
            input_payload = json.load(f)
    else:
        input_payload = convert_input_example_to_serving_input(embedded_input_example)

    model_config = json.loads(args.model_config_json)
    tags = dict(kv.split("=", 1) for kv in args.model_tag) if args.model_tag else {}

    # stage → save → load → predict → move
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write staged model
        mlflow.pyfunc.save_model(
            path=tmp_dir,
            python_model=ModelClass(),
            code_paths=["model_code"],
            artifacts=artifacts,
            model_config=model_config,
            signature=signature,
            input_example=embedded_input_example,  # always embed original input example
            pip_requirements="requirements.txt",
        )

        try:
            # Runs inference on the packaged model to ensure
            # the packaged model can actually run predictions
            validate_serving_input(model_uri=tmp_dir, serving_input=input_payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Forward-pass failed: %s", exc)
            sys.exit(1)

        # move into final location atomically
        dest = Path(args.out_dir)
        if dest.exists():
            LOGGER.error("Destination '%s' already exists.", dest)
            sys.exit(1)
        shutil.move(tmp_dir, dest)

    if tags:
        try:
            mlflow.set_tags(tags)
        except Exception:  # noqa: BLE001
            pass

    LOGGER.info(
        "✅ Model packaged & inference-validated → %s", os.path.abspath(args.out_dir)
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        sys.exit(1)
