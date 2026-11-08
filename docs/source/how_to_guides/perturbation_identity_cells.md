# Raw cells to labeled response panels

`czbenchmarks.datasets.perturbation_identity_cells` converts raw cell counts
into labeled donor-target response panels for `PerturbationIdentityTask`.
Model training and prediction are separate steps.

The official `PerturbationExpressionPrediction` task scores **cell-level model
predictions** against its DE truth using strict barcode mappings. Its estimand
is different. This adapter neither relabels DE truth nor manufactures cells,
predictions, reruns, or holdout declarations. The official CLI lists tasks; this
example does not introduce a benchmark-run CLI or claim VCP support.

## Input contract

Call `aggregate_anndata_responses(adata, donor_key=..., target_key=...,
control_key=..., control_value=..., context_key=..., target_catalog=...,
scored_genes=..., clone_key=None, count_layer=None,
expected_normalization_genes=None, chunk_size=1024)`.

- Supply one biological context, explicit donor labels, perturbation labels,
  control metadata, a **full prespecified target catalog**, and scored genes.
  IDs are exact, nonempty strings. Cell and gene IDs and requested catalogs
  must be unique. Donors are never inferred from cell barcodes.
- `control_key` can equal `target_key`, with `control_value="NTC"`, for example.
  Alternatively use a string or boolean control column and matching value.
  Control target labels cannot be scored targets or inconsistently annotated.
  Other metadata roles must use different column names.
- `clone_key` names the explicit clone/replicate column. Without it, cells are
  pooled by donor. Clones are scoped to their donor; identical clone names
  across donors are valid. All donor/clone groups in the input participate,
  even control-only groups and groups with only excluded perturbations.
- Supply raw UMI counts in `X`, or an explicitly named raw-count layer. Every
  count, including excluded-target cells, is checked: finite, nonnegative,
  integral and below the source floating-point integer-precision limit.
  Conservatively reject total UMIs at or above 2^53 before aggregation. Zero
  cell libraries are rejected rather than silently dropped. X with `uns.log1p`
  is rejected; an explicit count layer bypasses that X-specific marker.
- Matrix checks **cannot prove raw-data provenance**. Integer-valued normalized
  data without identifying metadata can be indistinguishable from counts.
  Checks also **cannot prove a complete gene-expression feature universe**.
  Establish both from source documentation. The optional expected gene list
  enforces exact set equality to the supplied feature labels, not provenance.
  Do not pass an already filtered scored-gene-only matrix as full expression.
- Missing requested genes are an error, never a silent intersection. Targets
  outside the catalog are excluded explicitly and counted by target in the
  manifest. Controls must be present in every participating donor/clone.

## Transformation and coverage

Within each donor × clone × target (and matched control), pool raw UMIs and
normalize the **pooled library using all provided features**. Retain scored gene
coordinates, calculate `log1p(CP10k)`, subtract that clone's matched control, and
average clone responses equally. This is neither a mean of per-cell log values
nor an average weighted by clone cell counts.

The output `ResponsePanel` retains every donor × requested target combination.
Donors are sorted; targets and genes retain requested order. A donor-target with
any missing participating clone-target is unavailable: its entire response is
NaN and its minimum clone-target cell count is zero. Controls are never filled
with zeros, borrowed across clones, or synthesized.

- `units`: labeled `(donor, target)` MultiIndex; `genes`: gene Index.
- `responses`: rows × scored genes; `available`: complete clone-target coverage.
- `min_clone_cell_counts`: minimum **target** cell count over that donor's clones.
- `total_cell_counts`: target cell total over clones, including incomplete units.
- `clone_coverage`: number of represented target clones, not a fraction.
- `clone_counts`: explicit donor/clone/control-or-target cell counts and total
  pooled UMIs; this includes zero-count missing targets. Use control rows here
  to impose your own control-cell eligibility thresholds.
- `manifest`: metadata mapping, context, source layer, scale, full denominator
  gene list, catalog, audit counts and participating clone counts.

`panel.to_labeled_responses(mask)` requires an explicit nonempty boolean mask
and rejects unavailable selected rows. It returns the existing `LabeledResponses`
without changing any scorer. Minimum count thresholds, donor completeness,
holdouts and model comparison remain caller policy. Availability is not a claim
of sufficient statistical power or independent biological replicates.

## H5AD / Hydra loading and storage

`PerturbationIdentityCellsDataset` implements the official abstract `Dataset`
interface and can be instantiated with `load_custom_dataset` or Hydra using:

```yaml
_target_: czbenchmarks.datasets.perturbation_identity_cells.PerturbationIdentityCellsDataset
path: /path/to/raw_cells.h5ad
organism: ${organism:HUMAN}
donor_key: donor
target_key: target
control_key: target
control_value: NTC
context_key: context
clone_key: clone
target_catalog: [A, B]
scored_genes: [ENSG000001, ENSG000002]
chunk_size: 1024
```

Use `Organism.HUMAN` in a Python configuration (see the runnable example).
The bounded H5AD loader requires **AnnData >= 0.11** (tested with 0.11.4),
and raises a readable runtime error if its public I/O APIs are unavailable.
This extension does not change the base package dependency floor.
`load_data()` opens H5AD read-only, reads metadata and the requested expression
source, aggregates, and closes its handle on success **or failure**. It does not
retain `adata`, and is not a `SingleCellDataset` for cell-level native tasks.
`store_task_inputs()` requires a successfully loaded panel and writes numeric
NPZ (`allow_pickle=False` is sufficient to read), JSON labels and manifest, and
CSV clone counts. It does not serialize predictions or executable objects.

Dense and in-memory CSR/CSC matrices and backed dense/CSR H5AD X/layers are
processed by row chunks. **Backed CSC is rejected before count reads** because
AnnData row slicing can read its entire data/indices arrays. Convert CSC H5AD
to CSR in a separate, explicitly memory-budgeted preprocessing step, or use an
in-memory CSC matrix if it fits your budget. This adapter never silently loads
a full backed CSC matrix. Peak expression memory is proportional to `chunk_size × input genes`;
pooled state is proportional to observed donor/clone/target groups × scored
genes. Metadata and output remain in memory. Smaller chunks trade speed for
memory. An externally supplied backed AnnData may already have eager layers;
the Dataset H5AD loader avoids that AnnData loading behavior. No caller-owned
AnnData data, indices, metadata, sparse structures, or file handles are modified.

See `examples/example_perturbation_identity_cells.py` for a synthetic, explicitly
labeled interface demonstration. It is not biological validation.
