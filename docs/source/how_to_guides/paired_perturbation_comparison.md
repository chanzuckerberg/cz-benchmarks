# Paired perturbation response comparisons

`paired_mse_reduction` compares a model with explicit reference predictions on
exactly the same outcome rows and ordered genes. It adds a baseline comparison
without changing the existing masked-gene perturbation task.

## Definition and interpretation

For row i and G genes, let

- L_model(i) = (1/G) sum_g (y_true[i,g] - y_pred[i,g])^2
- L_reference(i) = (1/G) sum_g (y_true[i,g] - y_reference[i,g])^2
- reduction(i) = L_reference(i) - L_model(i).

Without `group_ids`, the reported score is the mean of reduction(i) over rows.
With labels identifying D groups, it is (1/D) sum_d mean_{i in d} reduction(i).
Genes and groups receive equal weights; rows within a group receive equal weights.
Positive values favor the model. Units are squared expression-response units.
A worse model has a negative score. Identical predictions score zero.

Use one row per explicitly aggregated donor/stratum/perturbation unit if this
matches the scientific design. Passing donor IDs balances donors, but does not
validate donor holdout, control matching, replicate independence, or preprocessing.
For several strata, aggregation within donors remains row weighted; choose and
report a prespecified target population instead of silently implying equal strata.

## Example

```python
import numpy as np
from czbenchmarks.metrics import metrics_registry
from czbenchmarks.metrics.types import MetricResult, MetricType

truth = np.eye(2)
prediction = truth.copy()
generic_reference = np.full((2, 2), 0.5)
value = metrics_registry.compute(
    MetricType.PAIRED_MSE_REDUCTION,
    y_true=truth,
    y_pred=prediction,
    y_reference=generic_reference,
    group_ids=["donor_a", "donor_b"],
)
result = MetricResult(
    metric_type=MetricType.PAIRED_MSE_REDUCTION,
    value=value,  # 0.25, a synthetic arithmetic example
    params={"reference": "generic", "aggregation": "equal_donor"},
)
```

A reference can be an identity-agnostic prediction trained only on training
outcomes, or predictions obtained by rerunning a model with matched alternative
perturbation inputs. Those references test different hypotheses and should be
reported separately. Swapping already-computed predictions tests alignment of
response vectors with labels, not whether the model consumes an identity input.

## Scientific safeguards

- All matrices must have identical row identities, gene identities, order, and
  response scale. Shape checking cannot establish semantic alignment. Values
  must be finite; masked values are rejected rather than silently ignored.
- Fit preprocessing, feature selection, generic responses and hyperparameters
  using training donors only. Construct test outcomes using prespecified,
  biologically matched control samples. Declare any test-time control information.
- Keep donor, batch, cell type and other design constraints when constructing
  alternative identities. Do not shuffle cells as independent biological samples.
- Do not apply direct row swaps to condition-specific DE vectors from the existing
  perturbation task: these may contain different gene coordinates and lengths.
- Report the absolute model and reference losses alongside the gain, plus the
  reference construction, split, response scale, gene panel and weighting.
- A reference-relative gain is predictive evidence conditional on that reference.
  It is not conditional mutual information, causal identification, proof of
  mechanistic understanding, or assurance that a model uses perturbation identity.
- Zero gain can reflect a weak perturbation effect or a poor identity-aware model;
  a strong generic reference is necessary for an informative comparison.
- This scalar has no uncertainty or significance test built in. Report paired
  uncertainty at the biological replication level, and avoid treating cells as
  independent donors. Very small donor counts do not support precise population
  inference. Comparisons must use the same target set and weights.

The existing Spearman metric remains useful for per-condition response direction;
this loss difference provides a complementary, scale-sensitive comparison with a
specified reference. It is not an automatic replacement for existing metrics.

## Labeled identity-input comparison

For preaggregated donor-target response matrices, the optional
`compare_identity_predictions` adapter aligns each submitted prediction to truth
by explicit donor, target and gene labels before computing equal-donor MSEs and
per-donor differences. It requires identical sets of units and genes across all
three matrices, matching response-scale declarations, and a declared identity
input for every prediction row. Use one prespecified context or stratum per call;
the two-level index cannot distinguish repeated donor-target pairs across contexts.
It refuses duplicate or missing labels and rows
where the alternative identity equals the evaluated target. The adapter does
not select a gene intersection, match controls, fit preprocessing, or run a
model. A matching scale string cannot prove matching biological transformations.

```python
import numpy as np
import pandas as pd
from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
    LabeledResponses, IdentityPredictions, compare_identity_predictions,
)

units = pd.MultiIndex.from_tuples([("donor1", "targetA"), ("donor1", "targetB")])
genes = pd.Index(["gene1", "gene2"])
scale = "declared response transformation and controls"
truth = LabeledResponses(np.zeros((2, 2)), units, genes, scale)
matched = IdentityPredictions(
    LabeledResponses(np.zeros((2, 2)), units, genes, scale), ["targetA", "targetB"]
)
alternative = IdentityPredictions(
    LabeledResponses(np.ones((2, 2)), units, genes, scale), ["targetB", "targetA"]
)
result = compare_identity_predictions(truth, matched, alternative)
# This is a synthetic arithmetic example: result.paired_mse_reduction == 1.0.
```

Each alternative row must remain indexed by the original evaluated donor and
original target, even though the supplied identity differs. The alternative
array must come from a documented rerun with alternative identities if the
purpose is to study input sensitivity. The adapter cannot verify such a rerun
from arrays and labels. If the array instead consists of reassigned outputs, the
comparison tests response-label alignment and must be reported as such. No conclusion about
whether a model uses identity follows from the function call alone. These
preaggregated inputs are distinct from the existing task's cell-by-gene
reconstructions and its condition-specific, potentially different DE gene sets;
do not pass the latter as if they were a common response panel.

## Native donor-held-out response task

`PerturbationIdentityTask` is a separate registered CZI task, not an extension
of the existing cell-level perturbation-expression task. It takes a dense
`numpy.ndarray` of matched predictions as `cell_representation` through
`Task.run`, together with `PerturbationIdentityTaskInput` containing labeled
preaggregated donor-target truth, matching row and gene labels for the matched
matrix, a separately supplied labeled alternative-identity matrix and a
separately supplied labeled generic matrix. See
`examples/example_perturbation_identity_task.py` for a complete synthetic
registry, input-validation and execution example. The example's numbers are
not biological data or outputs from a model.

The task verifies identical donor-target and gene sets, response-scale
labels and identity declarations; aligns each prediction separately to truth;
requires a target-invariant generic output within each donor; and rejects
unlabeled intersections, masked values, nonfinite losses and unsupported sparse
or DataFrame inputs. One call represents one prespecified biological context.
Results include each model/reference absolute MSE, the alternative-minus-matched
and generic-minus-matched MSE differences, each by donor and with donors weighted
equally. Metric parameters include dataset, context, scale, a hash of the
labeled panel and prediction/reference IDs to keep incompatible results apart.

Provide declared fitted donor lists per evaluation donor for matched,
alternative, generic, and all learned preprocessing components. Matched and
alternative must declare identical fitted donors. Each declaration must exclude
the corresponding evaluation donor. Empty learned-preprocessing lists are
allowed when no preprocessing is learned. Declare permitted use of held-out
controls separately. These checks reject a contradictory manifest; they cannot
prove that training, nested tuning, gene selection or transformations actually
followed it. Document split manifests, controls, model provenance and how the
alternative was generated outside the task. For an input-sensitivity comparison,
provide real predictions obtained by changing only the test identity input.
Output reassignment must be marked separately and does not establish model-input
use. Matching labels, target-invariant generic outputs and identical scale
strings likewise do not prove absence of leakage, identical context or a
biologically valid design. No uncertainty or population inference is supplied.
