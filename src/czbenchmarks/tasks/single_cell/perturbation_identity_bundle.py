"""Portable, pickle-free inputs for the native supplied-prediction task.

Hashes bind file contents, not biological validity, data rights, or truthful
model execution declarations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from ...metrics.types import MetricResult
from .perturbation_identity_response import IdentityPredictions, LabeledResponses
from .perturbation_identity_task import (
    PerturbationIdentityTask,
    PerturbationIdentityTaskInput,
)

SCHEMA = "czbenchmarks.perturbation_identity_response.bundle.v1"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MATRICES = ("truth", "matched", "alternative", "generic")
ARRAY_NAMES = frozenset(
    [
        f"{m}_{field}"
        for m in MATRICES
        for field in ("values", "donors", "targets", "genes")
    ]
    + ["matched_identity_inputs", "alternative_identity_inputs"]
)
TEXT_FIELDS = (
    "dataset_id",
    "context",
    "matched_model_id",
    "alternative_reference_id",
    "generic_reference_id",
    "heldout_control_usage",
    "alternative_kind",
    "alternative_provenance",
    "generic_provenance",
)
METADATA_FIELDS = frozenset((*TEXT_FIELDS, "fitted_donors"))


@dataclass(frozen=True)
class PredictionBundle:
    """Structurally validated input; score with the native task for semantic checks."""

    matched_values: np.ndarray
    task_input: PerturbationIdentityTaskInput
    sha256: str

    def score(self) -> list[MetricResult]:
        """Apply the unchanged native task's alignment, holdout and loss checks."""
        return PerturbationIdentityTask().run(self.matched_values, self.task_input)


def _json(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid manifest JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Manifest must be a JSON object")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _metadata(manifest):
    _keys(manifest, {"schema", "metadata", "scales", "array_sha256"}, "manifest")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported bundle schema version")
    metadata = manifest["metadata"]
    _keys(metadata, METADATA_FIELDS, "metadata")
    for field in TEXT_FIELDS:
        if type(metadata[field]) is not str or not metadata[field].strip():
            raise ValueError(f"{field} must be a nonempty JSON string")
    _keys(manifest["scales"], MATRICES, "scales")
    if any(type(v) is not str or not v.strip() for v in manifest["scales"].values()):
        raise ValueError("Response scales must be nonempty JSON strings")
    _keys(manifest["array_sha256"], ARRAY_NAMES, "array_sha256")
    if any(
        type(v) is not str or not re.fullmatch(r"[0-9a-f]{64}", v)
        for v in manifest["array_sha256"].values()
    ):
        raise ValueError("Array digests must be lowercase SHA-256 strings")
    fitted = metadata["fitted_donors"]
    _keys(
        fitted,
        {"matched", "alternative", "generic", "learned_preprocessing"},
        "fitted_donors",
    )
    for folds in fitted.values():
        if not isinstance(folds, dict):
            raise ValueError("Fitted donor declarations must be JSON objects")
        for donor, training in folds.items():
            if (
                type(donor) is not str
                or not donor.strip()
                or type(training) is not list
                or any(type(v) is not str or not v.strip() for v in training)
            ):
                raise ValueError(
                    "Fitted donor IDs must be lists of nonempty JSON strings"
                )
    return metadata


def _npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _read_npy(raw: bytes, name: str, limit: int) -> np.ndarray:
    """Check declared allocation before NumPy sees a possibly hostile header."""
    buffer = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(buffer)
        else:
            raise ValueError("Only NPY versions 1 and 2 are supported")
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.kind not in "fiuU"
            or dtype.itemsize == 0
        ):
            raise ValueError(f"{name}: unsupported or object dtype")
        if len(shape) > 2 or any(type(v) is not int or v < 0 for v in shape):
            raise ValueError(f"{name}: invalid array shape")
        elements = math.prod(shape)
        if elements > limit:
            raise ValueError(f"{name}: logical element count exceeds size limit")
        size = elements * dtype.itemsize
        if size > limit or size != len(raw) - buffer.tell():
            raise ValueError(
                f"{name}: declared array size exceeds limit or mismatches payload"
            )
        # Never enable pickle, even for a digest-verified archive.
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except (EOFError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: malformed NPY array") from exc


def _check_arrays(arrays):
    for name, array in arrays.items():
        if name.endswith("_values"):
            if (
                array.ndim != 2
                or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} must be a finite real numeric matrix")
        elif (
            array.ndim != 1
            or array.dtype.kind != "U"
            or any(not value.strip() for value in array.tolist())
        ):
            raise ValueError(f"{name} must be a vector of nonempty Unicode strings")
    for matrix in MATRICES:
        shape = arrays[f"{matrix}_values"].shape
        if (
            shape != (len(arrays[f"{matrix}_donors"]), len(arrays[f"{matrix}_genes"]))
            or len(arrays[f"{matrix}_targets"]) != shape[0]
        ):
            raise ValueError(f"{matrix}: values and labels have different shapes")
    for matrix in ("matched", "alternative"):
        if len(arrays[f"{matrix}_identity_inputs"]) != len(arrays[f"{matrix}_donors"]):
            raise ValueError(f"{matrix}: one identity input is required per row")


def load_prediction_bundle(
    path, *, expected_sha256=None, max_bytes=DEFAULT_MAX_BYTES
) -> PredictionBundle:
    """Read one bounded NPZ snapshot, with no extraction, pickle or type coercion.

    Internal digests detect inconsistent edits. A separately trusted whole-file
    hash also detects rewritten data/metadata; neither authenticates provenance.
    The byte limit is a resource bound, not a guarantee about total peak RAM.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if expected_sha256 is not None and (
        type(expected_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 string")
    with open(path, "rb") as handle:
        snapshot = handle.read(max_bytes + 1)
    if len(snapshot) > max_bytes:
        raise ValueError("Archive exceeds max_bytes")
    digest = hashlib.sha256(snapshot).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Archive SHA-256 does not match expected_sha256")
    expected = {f"{name}.npy" for name in ARRAY_NAMES | {"manifest"}}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            members = archive.infolist()
            if (
                len(members) != len(expected)
                or {m.filename for m in members} != expected
            ):
                raise ValueError("Archive has missing, duplicate or unexpected members")
            if sum(m.file_size for m in members) > max_bytes:
                raise ValueError("Uncompressed archive exceeds max_bytes")
            raw = {}
            for member in members:
                if member.flag_bits & 1 or member.compress_type not in (
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                ):
                    raise ValueError("Encrypted or unsupported ZIP compression")
                limit = (
                    min(max_bytes, MAX_MANIFEST_BYTES)
                    if member.filename == "manifest.npy"
                    else max_bytes
                )
                if member.file_size > limit:
                    raise ValueError(f"{member.filename} exceeds size limit")
                with archive.open(member) as stream:
                    value = stream.read(limit + 1)
                if len(value) > limit or len(value) != member.file_size:
                    raise ValueError("ZIP member size mismatch")
                raw[member.filename[:-4]] = value
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("Invalid prediction bundle ZIP") from exc
    manifest_array = _read_npy(raw.pop("manifest"), "manifest", MAX_MANIFEST_BYTES)
    if manifest_array.ndim != 0 or manifest_array.dtype.kind != "U":
        raise ValueError("Manifest must be a scalar Unicode JSON string")
    manifest = _json(manifest_array.item())
    metadata = _metadata(manifest)
    arrays = {}
    for name, value in raw.items():
        if hashlib.sha256(value).hexdigest() != manifest["array_sha256"][name]:
            raise ValueError(f"{name}: array SHA-256 mismatch")
        arrays[name] = _read_npy(value, name, max_bytes)
    _check_arrays(arrays)
    responses = {
        m: LabeledResponses(
            arrays[f"{m}_values"],
            pd.MultiIndex.from_arrays(
                [arrays[f"{m}_donors"], arrays[f"{m}_targets"]],
                names=["donor", "target"],
            ),
            pd.Index(arrays[f"{m}_genes"]),
            manifest["scales"][m],
        )
        for m in MATRICES
    }
    inputs = PerturbationIdentityTaskInput(
        truth=responses["truth"],
        matched_units=responses["matched"].units,
        matched_genes=responses["matched"].genes,
        matched_scale=responses["matched"].scale,
        matched_identity_inputs=arrays["matched_identity_inputs"].tolist(),
        alternative=IdentityPredictions(
            responses["alternative"], arrays["alternative_identity_inputs"].tolist()
        ),
        generic=responses["generic"],
        **metadata,
    )
    return PredictionBundle(responses["matched"].values, inputs, digest)


def _unicode_labels(labels):
    values = list(labels)
    if any(not isinstance(value, str) or "\x00" in value for value in values):
        raise ValueError("Portable labels must be strings without NUL characters")
    return np.asarray(values, dtype="U")


def _write_new_file(destination, payload):
    """Publish a complete file without following or replacing an existing path."""
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def save_prediction_bundle(
    path, matched_values, task_input, *, max_bytes=DEFAULT_MAX_BYTES
) -> str:
    """Validate and save a new archive atomically without replacing any file.

    Numeric values are stored as float64, the native scoring precision. Row and
    gene orders remain independent across all four matrices. No predictions are
    generated or permuted by this function.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    PerturbationIdentityTask().run(matched_values, task_input)
    responses = {
        "truth": task_input.truth,
        "matched": LabeledResponses(
            matched_values,
            task_input.matched_units,
            task_input.matched_genes,
            task_input.matched_scale,
        ),
        "alternative": task_input.alternative.responses,
        "generic": task_input.generic,
    }
    arrays = {}
    for name, response in responses.items():
        values = np.asarray(response.values)
        if values.dtype.kind not in "fiu":
            raise ValueError("Portable response values must have real numeric dtypes")
        arrays[f"{name}_values"] = np.ascontiguousarray(values, dtype="<f8")
        for field, labels in (
            ("donors", response.units.get_level_values(0)),
            ("targets", response.units.get_level_values(1)),
            ("genes", response.genes),
        ):
            arrays[f"{name}_{field}"] = _unicode_labels(labels.tolist())
    arrays["matched_identity_inputs"] = _unicode_labels(
        task_input.matched_identity_inputs
    )
    arrays["alternative_identity_inputs"] = _unicode_labels(
        task_input.alternative.identity_inputs
    )
    _check_arrays(arrays)
    raw = {name: _npy_bytes(array) for name, array in arrays.items()}
    manifest = {
        "schema": SCHEMA,
        "metadata": {name: getattr(task_input, name) for name in METADATA_FIELDS},
        "scales": {name: response.scale for name, response in responses.items()},
        "array_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in raw.items()
        },
    }
    _metadata(manifest)
    raw["manifest"] = _npy_bytes(
        np.asarray(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    )
    if (
        len(raw["manifest"]) > MAX_MANIFEST_BYTES
        or sum(map(len, raw.values())) > max_bytes
    ):
        raise ValueError("Bundle exceeds configured size limits")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(raw):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, raw[name])
    payload = buffer.getvalue()
    if len(payload) > max_bytes:
        raise ValueError("Archive exceeds max_bytes")
    _write_new_file(path, payload)
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> int:
    """Score a local bundle into machine-readable JSON, without network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON file; never overwritten"
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    bundle = load_prediction_bundle(
        args.bundle, expected_sha256=args.expected_sha256, max_bytes=args.max_bytes
    )
    results = bundle.score()
    report = {
        "bundle_schema": SCHEMA,
        "bundle_sha256": bundle.sha256,
        "task": "perturbation_identity_response",
        "task_module_sha256": hashlib.sha256(
            Path(__file__).with_name("perturbation_identity_task.py").read_bytes()
        ).hexdigest(),
        "bundle_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Supplied predictions; provenance declarations are not independently verified.",
        "metrics": [result.model_dump(mode="json") for result in results],
    }
    output = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    _write_new_file(args.output, output.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
