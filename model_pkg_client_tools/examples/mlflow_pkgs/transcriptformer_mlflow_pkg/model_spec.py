"""
Pure-Python specification for the **file_uri_to_tensor** archetype.

--- INSTRUCTIONS FOR CUSTOMIZATION ---
You must specify two things:

1) MODEL_SIGNATURE:

This step specifies the input and output schema for your model using
[MLflow ModelSignature data types](https://mlflow.org/docs/latest/ml/model/signatures):

   • The first input **must** be:
       ColSpec(type="string", name="input_uri")
     as your model stub relies on "input_uri" semantics.

   • To add extra required inputs, uncomment & edit the `ColSpec(...)` lines below.

   • To define your output tensor, uncomment & adjust the `TensorSpec(...)` line.

   • To add runtime parameters, uncomment & edit the `ParamSpec(...)` lines below.
       – **Note**: ParamSpec **requires** a `default=` value so MLflow can
         validate missing params and provide UIs with sensible defaults.

2) INPUT_EXAMPLE:

This step encodes a real input that will verify correctness of the generated
model package artifact by running the input through a forward pass. It also
provides a concrete example that will be stored with package and serve as documentation:

   • Must match the MODEL_SIGNATURE above.

   • Must be a **REAL INPUT** that can go through a forward pass.

   • Provides a concrete example for both input_example.json and serving_input_example.json.
"""

import numpy as np  # noqa: F401
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import (
    Schema,
    ColSpec,
    TensorSpec,  # noqa: F401
    ParamSchema,
    ParamSpec,  # noqa: F401
)

# -----------------------------------------------------------------------------
# 1) MODEL_SIGNATURE
# -----------------------------------------------------------------------------
MODEL_SIGNATURE = ModelSignature(
    inputs=Schema(
        [
            # REQUIRED FILE URI INPUT COLUMNS
            ColSpec(type="string", name="input_uri"),  # DO NOT REMOVE
            # **STEP 1.1:** ADD ONE OR MORE EXTRA REQUIRED COLUMNS (uncomment & edit)
            # ColSpec(type="string", name="gene_to_perturb"),
        ]
    ),
    outputs=Schema(
        [
            # **STEP 1.2:** ADD ONE OR MORE OUTPUT TENSOR SPECS (uncomment & edit)
            TensorSpec(type=np.dtype("float32"), shape=[-1, 2048]),
        ]
    ),
    params=ParamSchema(
        [
            # **STEP 1.3:** ADD ONE OR MORE OPTIONAL RUNTIME PARAMETERS (uncomment & edit)
            ParamSpec(
                name="batch_size",
                dtype="integer",
                default=-1,  # Sentinel value to trigger family specific heuristic
            ),
            ParamSpec(
                name="precision",
                dtype="string",
                default="16-mixed",
            ),
            ParamSpec(
                name="gene_col_name",
                dtype="string",
                default="ensembl_id",
            ),
            ParamSpec(
                name="pretrained_embedding",
                dtype="string",
                default="",
            ),
        ]
    ),
)

# -----------------------------------------------------------------------------
# 2) INPUT_EXAMPLE
# -----------------------------------------------------------------------------
# * The INPUT_EXAMPLE must match with the MODEL_SIGNATURE above.
# * The INPUT_EXAMPLE **must be a REAL input** that can run through a forward pass.
#   This gives you a quick smoke test to check that the packaged model
#   can run. We suggest you pick the smallest possible example to serve as a smoke test.
INPUT_EXAMPLE: tuple[pd.DataFrame, dict] = (
    pd.DataFrame(
        {
            "input_uri": ["/home/ssm-user/.cz-benchmarks/datasets/example_small.h5ad"],
        }
    ),
    {
        # Provide key-value pairs matching your ParamSpec. If you omit keys
        # mlflow will substitute the omitted keys with their default values
        # as specified in ParamSpec:
        "batch_size": 32,
        "precision": "16-mixed",
        "gene_col_name": "ensembl_id",
    },
)
