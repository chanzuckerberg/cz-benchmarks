"""Synthetic registry example; the arrays below are not model reruns or biological data."""

import numpy as np
import pandas as pd

from czbenchmarks.tasks import TASK_REGISTRY
from czbenchmarks.tasks.single_cell.perturbation_identity_response import (
    IdentityPredictions,
    LabeledResponses,
)

units = pd.MultiIndex.from_tuples(
    [("donor1", "a"), ("donor1", "b"), ("donor2", "a"), ("donor2", "b")]
)
genes = pd.Index(["g1", "g2"])
scale = "synthetic response scale"
truth = LabeledResponses(np.zeros((4, 2)), units, genes, scale)
matched = np.full((4, 2), 0.1)
alternative = IdentityPredictions(
    LabeledResponses(np.ones((4, 2)), units, genes, scale), ["b", "a", "b", "a"]
)
generic = LabeledResponses(np.full((4, 2), 0.5), units, genes, scale)
folds = {"donor1": ["donor2"], "donor2": ["donor1"]}
params = dict(
    truth=truth,
    matched_units=units,
    matched_genes=genes,
    matched_scale=scale,
    matched_identity_inputs=["a", "b", "a", "b"],
    alternative=alternative,
    generic=generic,
    fitted_donors={
        name: folds
        for name in ["matched", "alternative", "generic", "learned_preprocessing"]
    },
    dataset_id="synthetic-data",
    context="synthetic-context",
    matched_model_id="synthetic-model",
    alternative_reference_id="synthetic-alternative",
    generic_reference_id="synthetic-generic",
    heldout_control_usage="none; synthetic arithmetic example",
    alternative_kind="output_reassignment",
    alternative_provenance="synthetic numbers, not a model execution",
    generic_provenance="synthetic constant",
)
name = "perturbation_identity_response"
task_input = TASK_REGISTRY.validate_task_inputs(name, params)
task = TASK_REGISTRY.get_task_class(name)()
for result in task.run(matched, task_input):
    if result.params["aggregation"] == "equal_donor":
        print(result.params["measure"], result.value)
