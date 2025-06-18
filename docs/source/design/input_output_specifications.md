
# Input Output Specifications for Datasets, Models, Tasks and Metrics

## Motivation

Ensuring compatibility between datasets, models, tasks, and metrics is critical to avoid costly failures during benchmarking workflows. By surfacing problems early—before allocating compute resources or loading large datasets—users can address issues proactively, saving time and resources.

- **Fail-fast validation:** The specification enables early detection of incompatibilities, providing clear, human-readable error messages for any mismatch across datasets, models, tasks, and metrics.
- **CLI integration:** Supports a `validate` command-line option to check the compatibility of Datasets, Models, Tasks, and Metrics before execution.
- **Asset management:** Facilitates consistent asset management by ensuring that all registered assets conform to expected input/output specifications, reducing manual errors and improving reproducibility.
- **Improved user experience:** Users receive actionable feedback, making it easier to debug and resolve configuration issues prior to running benchmarks.



### `InputOutputSpecs`


```python
class InputOutputSpecs(BaseModel, extra="forbid"):
    name        : str                                 # Human-readable specification name
    modality    : str                                 # e.g. "rna", "dna", "image"
    semantic    : str                                 # Ontology reference (EDAM, OBI, GO, etc.)
    version     : str = "1.0.0"                       # Increment when axes or encoding expectations change
    axes        : Optional[List[Axis]] = None         # List of axes (e.g., cells, genes)
    formats     : List[str] = ["h5ad"]                # Preferred file formats, ordered by preference
    encodings   : Optional[List[str]] = ["dense"]     # Data encodings, e.g., "csr", "png"
    validator   : Optional[str] = None                # Python function for validation (python:module:function)
    validation_schema : Optional[Dict[str, Any]] = None # Optional schema for validator
    description : Optional[str] = None                # Brief description

    @property
    def key(self) -> str:
        """Canonical identifier used everywhere else."""
        return f"{self.formats[0]}{self.modality}:{self.semantic}:{self.version}"
```


### `DatasetSpec`


```python
class DatasetSpec(BaseModel, extra="forbid"):
    """
    Describes a dataset's structure, provenance, and associated specifications.
    """
    name         : str                                 # Human-readable dataset name
    description  : Optional[str] = None                # Brief description of the dataset
    specs        : List[InputOutputSpecs]           # Unique set of input/output specifications
    provenance   : Optional[Provenance] = None         # Dataset origin and authorship info
    checksum     : Optional[str] = None                # Optional data integrity checksum
    created_at   : Optional[datetime] = None           # Creation timestamp
    metadata     : Dict[str, Any] = {}                 # Arbitrary additional metadata
```


###  `ModelSpec`


```python
class ModelSpec(BaseModel, extra="forbid"):
    """
    Describes a runnable model, its inputs, outputs, and metadata.
    """
    name              : str                                 # Human-readable model name
    description       : Optional[str] = None                # Brief description of the model
    version           : str                                 # Model version
    required_inputs   : List[InputOutputSpecs] = []      # All required input specifications
    optional_inputs   : List[InputOutputSpecs] = []      # Optional input specifications (e.g., masks)
    outputs           :  List[InputOutputSpecs]  
    compute           : Resource = Resource()               # Compute resource requirements
    provenance        : Optional[Provenance] = None         # Model origin and authorship info
    metadata          : Dict[str, Any] = {}                 # Arbitrary additional metadata
    supported_tasks   : Optional[List[str]] = None          # List of supported TaskSpec names/IDs
    supported_metrics : Optional[List[str]] = None          # List of supported MetricSpec names/IDs
```



### `MetricSpec`

```python
class MetricSpec(BaseModel, extra="forbid"):
    """
    Describes a metric used to evaluate model performance.
    """
    name        : str                                 # Human-readable metric name
    description : Optional[str] = None                # Brief description of the metric
    category    : str                                 # Metric type/category (e.g., "clustering", "classification")
    impl        : str                                 # Python function implementing the metric (python:module:function)
    direction   : Optional[Literal["higher_better", "lower_better"]] = None
                                                     # Indicates if higher or lower values are better
    inputs      : List[InputOutputSpecs]                           # List of InputOutputSpecs (expected model outputs and references)
    outputs     : Optional[List[InputOutputSpecs]] = None          # Optional list of InputOutputSpecs (metric output schemas, e.g., "score")
    range       : Tuple[float, float] = (0.0, 1.0)    # Expected value range for the metric
    threshold   : Optional[float] = None              # Optional threshold for pass/fail
    metadata    : Dict[str, Any] = {}                 # Arbitrary additional metadata
```




###  `TaskSpec`


```python
class TaskSpec(BaseModel, extra="forbid"):
    name               : str
    goal               : Literal[ # type
        "clustering", "classification",
        "dimensionality_reduction", "perturbation",
        "spatial_registration", "custom"
    ]                                         # Task goal/type
    input_schema       : InputOutputSpecs   # Model input schema
    prediction_schema  : InputOutputSpecs   # Model output schema
    reference_schema   : InputOutputSpecs   # Ground truth schema from dataset
    metrics            : list[MetricSpec]      # List of metrics to evaluate
    context            : dict[str, Any] = {}   # Organism, tissue, etc.
    metadata           : dict[str, Any] = {}   # Additional metadata
```


## Optionals


### Axis, Provenance, Resource

```py
# Add and validate dimensions od datasest if applicabale
class Axis(BaseModel, extra="forbid"):
    name : str                         # "cells", "genes", "x", "time", …
    size : Optional[int] = None        # None = variable
    kind : Literal[
        "sample", "feature", "spatial", "channel",
        "sequence", "time", "batch", "custom"
    ] = "custom"

# Dataset Provenance. May be useful in cli for search capabilities and VCP integration
class Provenance(BaseModel, extra="forbid"):
    source   : Literal["internal", "external"] = "external"
    version  : str
    authors  : Optional[List[str]]
    license  : Optional[str]
    citation : Optional[str]
    url      : Optional[HttpUrl]
    doi      : Optional[str]


# Useful to validate hardware resource requirement for a model
class Resource(BaseModel, extra="forbid"):
    gpu_type       : str
    gpu            : bool = False
    gpu_mem_mb     : Optional[int]
    cpu_cores      : int = 4
    ram_mb         : int = 8_192
    disk_mb        : int = 10_240
    cloud_provider : Optional[Literal["aws", "gcp", "azure", "custom"]] = None
    instance_type  : Optional[str] = None  # e.g., "n1-standard-8", "m5.large"
    region         : Optional[str] = None  # e.g., "us-west-2"
    validate_cloud : Optional[bool] = False

    def validate(self) -> bool:
        """
        Validates if the specified cloud instance type meets the resource requirements.
        Returns True if valid, False otherwise.
        """
        # Placeholder for actual cloud API validation logic
        # In production, query the provider for instance specs and compare

```

A key can be added to spec to uniquly identify the input output specification.

```py
    @property
    def key(self) -> str:        # canonical identifier used everywhere else
        return f"{self.format}{self.modality}:{self.semantic}:{self.version}"
```


###  `CompatibilityIssue` (Optional)

A human-readable item explaining one incompatibility.

```python
class CompatibilityIssue(BaseModel):
    who     : Literal["dataset","model","task","metric"]
    code    : str          # e.g. "MISSING_INPUT", "FORMAT_MISMATCH"
    message : str          # plain English explanation
```



### `CompatibilityChecker`

It inspects a dataset, a model and a task and returns compatibility problems it finds.

```python
class CompatibilityChecker:
    """Fail-fast validation of Dataset × Model × Task combinations."""

    @staticmethod
    def check_compatibility(
        dataset : DatasetSpec,
        model   : ModelSpec,
        task    : TaskSpec
    ) -> list[CompatibilityIssue]:
        issues : list[CompatibilityIssue] = []

        # ── helper: index dataset specifications ───────────────────────
        ds_specifications = {s.key: s for s in dataset.schemas}

        # 1️⃣ Dataset provides every required model input
        for need in model.required_in:
            if need.key not in ds_specifications:
                issues.append(CompatibilityIssue(
                    who="dataset",
                    code="MISSING_INPUT",
                    message=f"Dataset lacks required specification '{need.key}' "
                            f"needed by model '{model.name}'"
                ))
            else:
                ds_fmt = ds_specifications[need.key].formats
                if not set(ds_fmt) & set(need.formats):
                    issues.append(CompatibilityIssue(
                        who="dataset",
                        code="FORMAT_MISMATCH",
                        message=f"specification '{need.key}' present but formats "
                                f"do not overlap (dataset {ds_fmt} vs "
                                f"model {need.formats})"
                    ))

        # 2️⃣ Model indeed produces what the task expects
        if task.prediction.key not in [o.key for o in model.outputs]:
            issues.append(CompatibilityIssue(
                who="model",
                code="MISSING_OUTPUT",
                message=f"Model does not emit prediction specification "
                        f"'{task.prediction.key}' required by task "
                        f"'{task.name}'"
            ))

        # 3️⃣ Dataset carries the reference required by the task
        if task.reference.key not in ds_specifications:
            issues.append(CompatibilityIssue(
                who="dataset",
                code="MISSING_REFERENCE",
                message=f"Dataset lacks reference specification "
                        f"'{task.reference.key}' required by task "
                        f"'{task.name}'"
            ))

        # 4️⃣ Each metric can find its declared inputs
        supplied = {task.prediction.key, task.reference.key}
        for metric in task.metrics:
            needed = {i.key for i in metric.inputs}
            if not needed.issubset(supplied):
                missing = needed - supplied
                issues.append(CompatibilityIssue(
                    who="metric",
                    code="METRIC_INPUT",
                    message=f"Metric '{metric.name}' cannot find specifications "
                            f"{sorted(missing)}"
                ))
        return issues
```



### Run the checker before scheduling

```python
issues = CompatibilityChecker.check_compatibility(dataset, model, task)

if issues:
    print("❌  Incompatible combinations:")
    for i in issues:
        print(f"  • [{i.who}] {i.code} – {i.message}")
else:
    print("✅  All good, launch benchmark!")
```

#### Typical outputs


```
❌  Incompatible combo:
  • [dataset] MISSING_REFERENCE – Dataset lacks reference specification 'rna:cluster_labels:1.0' required by task 'Cell-clustering'
```

## Example usage of InputOutputSpecs

Define a specification

```py

# Example: Defining a specification for RNA data
rna_specification = InputOutputSpecs(
    name="RNA Expression Data",
    modality="rna",
    semantic="EDAM:1234",
    axes=["cells", "genes"],
    formats=["h5ad", "csv"],
    encodings=["dense", "csr"],
    description="specification for RNA expression data in h5ad or csv format."
)

print(rna_specification)
print("specification Key:", rna_specification.key)
```

Use specification to define Dataset specification

```py

# Example: Defining a dataset specification
rna_dataset_spec = DatasetSpec(
    name="Single-Cell RNA Dataset",
    description="A dataset containing single-cell RNA expression data.",
    specifications=[rna_specification],
    provenance={"source": "external", "doi": "10.1234/example"}, 
    created_at=datetime.now(),
    metadata={"organism": "human", "tissue": "liver"}
)

print(rna_dataset_spec)
```

Define a model spec

```py

# Example: Defining a model specification
rna_model_spec = ModelSpec(
    name="RNA Clustering Model",
    description="A model for clustering single-cell RNA expression data.",
    supported_inputs=[rna_specification],
    supported_outputs=[
        InputOutputSpecs(
            name="Cluster Labels",
            modality="rna",
            formats=["ndarray"],
            description="Cluster labels for RNA data."
        )
    ],
    supported_tasks=["Cell Clustering"],
    metadata={"framework": "PyTorch", "version": "1.0"}
)

print(rna_model_spec)
```

Define a task spec

```py

# Example: Defining a task specification
rna_task_spec = TaskSpec(
    name="Cell Clustering",
    description="A task for clustering single-cell RNA expression data.",
    required_inputs=[
        InputOutputSpecs(
            name="RNA Expression Data",
            modality="rna",
            semantic="EDAM:1234",
            axes=["cells", "genes"],
            formats=["h5ad", "csv"],
            encodings=["dense", "csr"],
            description="specification for RNA expression data in h5ad or csv format."
        )
    ],
    required_outputs=[
        InputOutputSpecs(
            name="Cluster Labels",
            modality="rna",
            formats=["ndarray"],
            description="Cluster labels for RNA data."
        )
    ],
    metadata={"task_type": "clustering"}
)

print(rna_task_spec)
```

Run Compatibility Checker

```py
# Example: Running the CompatibilityChecker with a task
issues = CompatibilityChecker.check_compatibility(
    dataset=rna_dataset_spec,  # Dataset defined earlier
    model=rna_model_spec,      # Model defined earlier
    task=rna_task_spec         # Task defined above
)

if issues:
    print("❌ Compatibility Issues Found:")
    for issue in issues:
        print(f"  - [{issue.who}] {issue.message}")
else:
    print("✅ All good! Dataset, model, and task are compatible.")
```

> **Note:** The code examples provided are for conceptual illustration and design purposes only and ocde have not been validated or tested.

---


# Declarative Validation (Nice to have)

## Motivation

- Enables a **declarative approach** for specifying dataset validation rules, making it easier to define and enforce data requirements without imperative code.
- Facilitates **integration of new models** by allowing model developers to declare expected input formats and validation logic, streamlining onboarding and reducing manual errors.
- Improves **interoperability and reproducibility** by standardizing how datasets are described and validated across different models and benchmarking workflows.
- Supports **automation and scalability** in benchmarking pipelines by allowing validation logic to be programmatically generated and executed, ensuring datasets meet model requirements before inference.
- Enhances **transparency and documentation** by making validation criteria explicit and human-readable, aiding both users and developers in understanding data expectations in computational biology applications.

```yaml
validator:
  - id: "single_cell_anndata_example"
    type: "AnnData"
    properties:
      X:
        type: "ndarray"
        dtype: "float32"
        shape: [null, null]  # flexible dimensions
      obs:
        type: "pandas_dataframe"
        columns:
          sample_id:
            type: "string"
          cell_type:
            type: "string"
      var:
        type: "pandas_dataframe"
        columns:
          gene_id:
            type: "string"
          gene_name:
            type: "string"
      uns:
        type: "object"
        properties:
          clustering_algorithm:
            type: "string"

  - id: "zarr_example"
    type: "Zarr"
    properties:
      data:
        type: "ndarray"
        dtype: "float32"
        shape: [null, null, null]  # e.g., spatial transcriptomics data
      metadata:
        type: "object"
        properties:
          description:
            type: "string"
          organism:
            type: "string"
          sequencing_type:
            type: "string"
            enum: ["RNA-seq", "DNA-seq", "Proteomics"]

  - id: "image_dataset_example"
    type: "Image"
    properties:
      images:
        type: "list"
        items:
          type: "object"
          properties:
            file_path:
              type: "string"
            image_size:
              type: "ndarray"
              dtype: "uint8"
              shape: [null, null, 3]  # RGB images

  - id: "dna_seq_fastq_example"
    type: "FASTQ"
    properties:
      files:
        type: "list"
        items:
          type: "object"
          properties:
            file_path:
              type: "string"
            read_length:
              type: "integer"
            paired_end:
              type: "boolean"

  - id: "proteomics_example"
    type: "ProteomicsMatrix"
    properties:
      quantification_matrix:
        type: "ndarray"
        dtype: "float32"
        shape: [null, null]  # samples x proteins
      sample_metadata:
        type: "pandas_dataframe"
        columns:
          sample_id:
            type: "string"
          condition:
            type: "string"

type_definitions:
  AnnData:
    python_type: "anndata.AnnData"
    validation_method: "validate_anndata"

  Zarr:
    python_type: "zarr.core.Array"
    validation_method: "validate_zarr_dataset"

  Image:
    python_type: "str"  # file path
    validation_method: "validate_image"

  FASTQ:
    python_type: "str"  # file path
    validation_method: "validate_fastq"

  ProteomicsMatrix:
    python_type: "pandas.DataFrame"
    validation_method: "validate_proteomics_df"

  ndarray:
    python_type: "numpy.ndarray"
    allowed_dtypes: ["float32", "float64", "uint8", "int32", "int64"]

  pandas_dataframe:
    python_type: "pandas.DataFrame"
```

```py
import yaml
import numpy as np
import pandas as pd
import anndata
import os
from typing import Any, Dict


class BaseValidator:
    def __init__(self, schema_file: str):
        self.schema = self.load_schema(schema_file)
        self.type_validators = {
            "numpy.ndarray": self.validate_ndarray,
            "pandas.DataFrame": self.validate_pandas_df,
            "str": self.validate_str,
            "int": self.validate_int,
            "bool": self.validate_bool,
            "object": self.validate_object,
        }

    @staticmethod
    def load_schema(schema_file: str) -> Dict[str, Any]:
        with open(schema_file, 'r') as file:
            return yaml.safe_load(file)

    def validate(self, dataset_id: str, data: Dict[str, Any]):
        dataset_schema = next((d for d in self.schema['datasets'] if d['id'] == dataset_id), None)
        if not dataset_schema:
            raise ValueError(f"Dataset ID '{dataset_id}' not found in schema.")
        self.validate_properties(dataset_schema['properties'], data)

    def validate_properties(self, properties: Dict[str, Any], data: Dict[str, Any]):
        for key, prop_schema in properties.items():
            if key not in data:
                raise ValueError(f"Missing required field '{key}'")

            value = data[key]
            type_def = prop_schema['type']

            if type_def not in self.schema['type_definitions']:
                raise ValueError(f"Undefined type '{type_def}' in schema.")

            python_type = self.schema['type_definitions'][type_def]['python_type']
            validator_method = self.type_validators.get(python_type)

            if validator_method is None:
                raise NotImplementedError(f"Validator for type '{python_type}' not implemented.")

            validator_method(value, prop_schema)

    def validate_ndarray(self, value, schema):
        if not isinstance(value, np.ndarray):
            raise TypeError("Expected numpy.ndarray")
        if 'dtype' in schema and value.dtype != schema['dtype']:
            raise TypeError(f"Expected dtype {schema['dtype']}, got {value.dtype}")
        if 'shape' in schema:
            expected_shape = schema['shape']
            if len(expected_shape) != value.ndim:
                raise ValueError(f"Expected array of ndim {len(expected_shape)}, got {value.ndim}")

    def validate_pandas_df(self, value, schema):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")
        if 'columns' in schema:
            for col, col_def in schema['columns'].items():
                if col not in value.columns:
                    raise ValueError(f"Missing required column '{col}' in DataFrame")

    def validate_str(self, value, schema):
        if not isinstance(value, str):
            raise TypeError("Expected str")

    def validate_int(self, value, schema):
        if not isinstance(value, int):
            raise TypeError("Expected int")

    def validate_bool(self, value, schema):
        if not isinstance(value, bool):
            raise TypeError("Expected bool")

    def validate_object(self, value, schema):
        if not isinstance(value, dict):
            raise TypeError("Expected object (dict)")
        if 'properties' in schema:
            self.validate_properties(schema['properties'], value)


class AnnDataValidator(BaseValidator):
    def __init__(self, schema_file: str):
        super().__init__(schema_file)
        self.type_validators.update({
            "anndata.AnnData": self.validate_anndata,
        })

    def validate_anndata(self, value, schema):
        if not isinstance(value, anndata.AnnData):
            raise TypeError("Expected anndata.AnnData")
        self.validate_properties(schema['properties'], {
            "X": value.X,
            "obs": value.obs,
            "var": value.var,
            "uns": value.uns
        })

    def validate_from_file(self, dataset_id: str, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        adata = anndata.read_h5ad(file_path)
        self.validate(dataset_id, {
            "X": adata.X,
            "obs": adata.obs,
            "var": adata.var,
            "uns": adata.uns
        })
```

> **Note:** The code examples provided are for conceptual illustration and design purposes only and ocde have not been validated or tested.

# Appendix

## Difference between InputOutputSpecs and Declarative Validations


|**Aspect**|**InputOutputSpecs**|**Declarative Validations**|
|---|---|---|
|**Definition**|A programmatic, reusable Python class that defines the structure and expectations for inputs/outputs.|A YAML-based schema that declaratively specifies validation rules for datasets and models.|
|**Implementation**|Implemented using Python and Pydantic for type safety and validation.|Defined in YAML files, interpreted by a validation engine at runtime.|
|**Purpose**|To describe the expected properties of inputs/outputs (e.g., format, modality, encoding).|To validate datasets and models against predefined rules without requiring custom code.|
|**Validation Scope**|Focused on programmatically validating inputs/outputs of datasets, models, tasks, etc.|Focused on validating datasets and their properties in a declarative manner.|
|**Use Case**|Best suited for defining reusable specifications for datasets, models, tasks, and metrics.|Best suited for validating datasets against predefined schemas in a benchmarking pipeline.|
|**Integration**|Integrated directly into Python code and used in specifications like `DatasetSpec`.|Used as an external configuration file, interpreted by a validation engine.|
|**Examples**|Defines fields like `name`, `modality`, `formats`, and `encodings`.|Defines properties like `X`, `obs`, `var`, and their types (e.g., `ndarray`, `DataFrame`).|



## **Comparison of Declarative Validation vs. Python Code Validation**

| **Aspect**          | **Declarative Validation**                                                                 | **Python Code Validation**                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Ease of Use**     | Easier and faster to define and modify validation rules using YAML or JSON.                | Depends on developer experience                                                                               |
| **Readability**     | Human-readable and self-documenting; validation rules are explicit and easy to understand. | Validation logic can become complex and harder to read, especially for large datasets.                        |
| **Reusability**     | Validation schemas can be reused across multiple projects without modification.            | Reusability depends on how well the Python code is modularized and abstracted.                                |
| **Error Handling**  | Errors are descriptive and tied to schema violations, making debugging easier.             | Errors depend on how well the Python code is written; may require additional effort to make them descriptive. |
| **Integration**     | Works well with configuration-driven workflows and pipelines.                              | Requires embedding validation logic directly into the codebase.                                               |
| **Transparency**    | Validation criteria are explicit and standardized, improving reproducibility.              | Validation logic may be opaque if not well-documented or modularized.                                         |
| **Tooling Support** | Can leverage existing tools for schema validation (e.g., JSON Schema, YAML parsers).       | Relies on Python libraries and custom implementations for validation.                                         |

