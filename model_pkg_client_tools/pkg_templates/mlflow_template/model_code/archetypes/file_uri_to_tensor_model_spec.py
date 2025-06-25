"""
Pure-Python specification for the **file_uri_to_tensor** archetype.
This file drives both:
  1) Stub generation (populating REQUIRED_COLS in your PythonModel)
  2) Model packaging (supplying MODEL_SIGNATURE and INPUT_EXAMPLE to mlflow_packager.py)

See MLflow’s official ModelSignature docs for full details:
https://mlflow.org/docs/latest/ml/model/signatures

--- INSTRUCTIONS FOR CUSTOMIZATION ---
1) MODEL_SIGNATURE:
   • The first input **must** be:
       ColSpec(type="string", name="input_uri")
     as your model stub relies on "input_uri" semantics.
   • To add extra required inputs, uncomment & edit the `ColSpec(...)` lines below.
   • To add runtime parameters, uncomment & edit the `ParamSpec(...)` lines below.
       – **Note**: ParamSpec **requires** a `default=` value so MLflow can
         validate missing params and provide UIs with sensible defaults.
   • To define your output tensor, uncomment & adjust the `TensorSpec(...)` line.

2) INPUT_EXAMPLE:
   • Must match the MODEL_SIGNATURE above.
   • Must be a **REAL INPUT** that can go through a forward pass.
   • Provides a concrete example for both input_example.json and serving_input_example.json.

Once you’ve edited this file, your stub and packager will import it directly—
no extra CLI flags needed.
"""

import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import (
    Schema,
    ColSpec,
    ParamSchema,
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
            # TensorSpec(type=np.dtype("float32"), shape=[-1, 2048]),
        ]
    ),
    params=ParamSchema(
        [
            # **STEP 1.3:** ADD ONE OR MORE OPTIONAL RUNTIME PARAMETERS (uncomment & edit)
            # ParamSpec(
            #     name="batch_size",
            #     dtype="integer",
            #     default=32,                     # required: must supply a default
            # ),
            # ParamSpec(
            #     name="precision",
            #     dtype="string",
            #     default="16-mixed",            # required: must supply a default
            # ),
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
            "input_uri": ["path/to/example.h5ad"],
            # Example: If you added gene_to_perturb ColSpec above, uncomment here:
            # "gene_to_perturb": ["TP53"],
        }
    ),
    {
        # Provide key-value pairs matching your ParamSpec. If you omit keys
        # mlflow will substitute the omitted keys with their default values
        # as specified in ParamSpec:
        # "batch_size": 32,
        # "precision": "16-mixed",
    },
)
