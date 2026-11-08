# Portable supplied-prediction bundles

This module provides a versioned, pickle-free file input for the
`perturbation_identity_response` task. It does not change its scoring, train a
model, generate predictions or certify a model's execution history.

## Export and score

After constructing a valid `PerturbationIdentityTaskInput` and its independently
labeled matched prediction matrix as in `example_perturbation_identity_task.py`:

```python
from czbenchmarks.tasks.single_cell.perturbation_identity_bundle import (
    save_prediction_bundle,
    load_prediction_bundle,
)

sha256 = save_prediction_bundle("predictions.npz", matched, task_input)
bundle = load_prediction_bundle("predictions.npz", expected_sha256=sha256)
results = bundle.score()
```

Export runs the native task's full semantic validation before publishing a file.
Load checks the container, manifest, numeric matrices and labeled dimensions;
`score()` then runs the unchanged native alignment, scale, generic-invariance,
held-out-donor and alternative-identity checks. Always call `score()` before
interpreting an imported bundle. The Python object remains mutable: its digest
identifies original file bytes, not subsequent in-memory edits.

The library's runnable module also writes native `MetricResult` JSON:

```bash
python -m czbenchmarks.tasks.single_cell.perturbation_identity_bundle \
  predictions.npz --output results.json --expected-sha256 HASH_FROM_EXPORT
```

The output includes the input SHA-256, schema, scorer source hashes and native
metrics. It is written directly as JSON, without terminal-width formatting.
Paths must be local. Both export and result writing atomically create **new**
files and refuse to overwrite an existing file or symlink, including the input.
Use a different output filename for another run.

## Version 1 contract

One NPZ (ZIP) contains exactly 19 NPY members. Each matrix has its own labels;
there is no assumption that its order matches another matrix.

- `truth_values`, `matched_values`, `alternative_values`, `generic_values`:
  finite, two-dimensional real numeric arrays. Export writes little-endian
  float64, matching native score precision. Import also accepts integer and
  real floating arrays; it never accepts object, complex, boolean or structured
  numeric data.
- For each of `truth`, `matched`, `alternative`, `generic`: suffixes `_donors`,
  `_targets`, `_genes` are one-dimensional Unicode string arrays. Donor and
  target arrays label rows together, and genes label columns. IDs are never
  silently matched by position, decoded from bytes or coerced from numbers.
  Export rejects NUL-containing labels to prevent Unicode-array truncation.
- `matched_identity_inputs` and `alternative_identity_inputs`: Unicode arrays
  in their respective prediction matrix's own row order. Alternative rows still
  refer to the original evaluated outcome, never to alternative-target truth.
- `manifest`: a scalar Unicode JSON object. Exactly four top-level fields:
  `schema`, `metadata`, `scales`, `array_sha256`.

`schema` is `czbenchmarks.perturbation_identity_response.bundle.v1`. `scales`
contains one string for each of the four matrices. `metadata` contains all nine
text declarations of the native input (`dataset_id`, `context`,
`matched_model_id`, `alternative_reference_id`, `generic_reference_id`,
`heldout_control_usage`, `alternative_kind`, `alternative_provenance`,
`generic_provenance`) plus `fitted_donors`. Each fitted-donor component maps every
evaluation donor to its actual declared list of fitting donors, which may
include donors absent from the evaluation panel. Empty learned-preprocessing
lists remain valid. Additional or duplicate keys are rejected.

`array_sha256` maps each of the 18 data-array names to the SHA-256 of its complete
NPY member bytes. This binds dtype, shape, storage order and payload. A changed
array with an old digest is rejected. A separately trusted whole-file hash also
binds metadata; self-contained hashes alone cannot authenticate a rewritten
archive. Export is deterministic for the same normalized arrays and declarations.

## Input boundaries and reproducibility

The importer reads one bounded byte snapshot and hashes exactly those bytes. It
never extracts ZIP paths or enables pickle. Unexpected/missing/duplicate archive
members, duplicate JSON keys, nonfinite JSON constants, unsupported schemas,
object dtypes and inconsistent shapes are rejected. Declared NPY allocation
sizes, including zero-width dtype tricks, are checked before array allocation.

The default maximum is 256 MiB for the archive and the sum of uncompressed
members, with a separate 1 MiB manifest limit. Use `max_bytes` or `--max-bytes`
only for a larger trusted input with suitable available memory. This bounds
input size, not peak process memory: Python/NumPy, the framework, labels, hashes
and scoring arrays require additional memory. Run large jobs serially in
resource-limited environments.

Hashes and declarations do **not** establish truthful donor holdout, equivalent
model inputs, correct controls, biological independence, data redistribution
rights, uncertainty calibration or actual alternative-input execution. Retain
source data/model versions, the training and inference scripts, controls and
permitted-information protocol with a study. One context per bundle remains
required. Scientific claims and permissions are unchanged from the native task.
