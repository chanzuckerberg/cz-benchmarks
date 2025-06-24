
## ModelAdapter Design (C)

**Consider for v1.0**

This Model Adapter design leverages a strongly-typed `ModelConfig` object, defined as a Pydantic model, to encapsulate all model configuration parameters, resource requirements, and input/output specifications. By using structured configuration and validation, this approach ensures clarity, correctness, and early error detection, while enabling robust, extensible, and production-ready model adapters.


```py
"""
# New enhanced models.yaml Config file
models:
  - name: "example-model"
    version: "1.2.3"
    git-sha: "ff43a8b"
    model_description: "Demo image classifier"
    model_image_uri: "docker.io/example/model:1.2.3"
    model_config_uri: "https://storage.myorg.com/configs/example-1.2.3.yaml"
    model_dockerfile: "https://github.com/myorg/models/Dockerfile"
"""

from pathlib import Path
from typing import Dict, Any, List

import yaml


class ModelRegistry:
    """Tiny helper around `registry.yaml`."""

    def __init__(self, path: Path):
        self.data: Dict[str, List[Dict[str, Any]]] = yaml.safe_load(path.read_text())

    def find(self, name: str, version: str) -> Dict[str, Any]:
        for m in self.data["models"]:
            if m["name"] == name and m["version"] == version:
                return m
        raise KeyError(f"{name}:{version} not in registry")

    def load_config(self, config_path: Path) -> Dict[str, Any]: 
        """Load a configuration file from the given path."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        return yaml.safe_load(config_path.read_text())

from __future__ import annotations
import json, urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    # I/O
    input_paths: Path
    output_dir:  Path
    # model arguments or hyper-params etc.
    model_parameters:  Dict[str, Any]
    batch_size:  Optional[int] = None 

    # provenance Optional
    model_name:    str
    model_version: str
    model_variant : Optional[str] = None

    # resources (inline, Or can be a separate class)
    cpu:    int = Field(2, ge=1)
    cpu_memory: int = Field(8, ge=1)
    gpu:    int = Field(0, ge=0)
    gpu_memory:  Optional[int]
    gpu_properties: Optional[Dict[str, Any]] = None
    disk_space:  int  = 100        # GB

    # misc
    timeout: Optional[int] = None


    # ---------- validators ----------
    @validator("input_paths", each_item=True)
    def _in_exists(cls, p: Path) -> Path:
        if not p.exists():
            raise FileNotFoundError(p)
        return p.resolve()

    @validator("output_dir", pre=True)
    def _out_dir(cls, p: Path | str) -> Path:
        p = Path(p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- helper ----------
    @staticmethod
    def load_from_uri(uri: str) -> Dict[str, Any]:
        """Return dict from YAML or JSON at `uri` (http(s) or local path)."""
        raw = (urllib.request.urlopen(uri).read().decode()
               if uri.startswith(("http://", "https://"))
               else Path(uri).read_text())
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            return json.loads(raw)



class ModelAdapter(abc.ABC):


    def __init__(self, config: ModelConfig):
        self.config = config
        self.log    = logging.getLogger(self.__class__.__name__)


    # ── Optional public façade to similify model author’s life
    def run(self) -> Any:
        """
        Calls: validate → setup → preprocess → predict → postprocess.
        `cleanup()` is invoked regardless of success/failure.
        """
        self.log.info("⏩  run()")
        try:
            self.validate()
            self.setup()
            data    = self.preprocess()
            result  = self.predict(data)
            return  self.postprocess(result)
        finally:
            self.cleanup()
            self.log.info("✅  done")

    # ── extension points for subclasses  
    @abc.abstractmethod
    def validate(self, data: Any) -> None:
        """Extra invariants beyond Pydantic’s checks (optional)."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Download weights, warm up GPU, spawn server, etc."""

    @abc.abstractmethod
    def preprocess(self) -> Any:
        """Turn raw `input` into model-ready tensors/objects."""

    @abc.abstractmethod
    def predict(self, data: Any) -> Any:
        """Run the forward pass / remote call."""

    @abc.abstractmethod
    def postprocess(self, result: Any) -> Any:
        """Convert model output to desired shape / schema."""

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Free resources, close files, terminate child procs."""
```


### Advantages

- **Strong Typing:** Uses Pydantic for config validation and type safety.
- **Lifecycle Management:** Orchestrates all steps in the inference pipeline.
- **Resource Specification:** Explicit resource fields (CPU, GPU, RAM, disk).


### Example Implementation - DockerAdapter (WIP)

> **Note:** The following code is a conceptual example and has not been fully tested. It is intended to illustrate the design approach and may require further development and validation before use.

```python

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import docker
from docker.types import Mount

from .config import ModelConfig
from czbenchmarks.dataset import BaseDataset

# Constants for container mount points
CONTAINER_INPUTS_DIR = "/mnt/inputs"
CONTAINER_OUTPUTS_DIR = "/mnt/outputs"

class DockerAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        image_uri: str,
        dataset: BaseDataset,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config)
        self.image = image_uri
        self.env = extra_env or {}
        self.cli = docker.from_env()
        self.container = None
        self.dataset = dataset



```

**Extra enhancements to docker config:**


- `network`: Docker network to attach the container to.
- `user`: User to run as inside the container.
- `working_dir`: Working directory inside the container.
- `tmpfs`: Temporary file systems to mount.
- `shm_size`: Size of shared memory (/dev/shm).
- `mem_limit`: Memory limit for the container.
- `cpus`: Number of CPUs to allocate.
- `detach`: Run container in detached mode.
- `remove`: Remove container after run.
- `restart_policy`: Restart policy for the container.
- `log_config`: Logging configuration for the container.
- `privileged`: Run container in privileged mode.
- `capabilities`: Linux capabilities to add.
- `security_opt`: Security options.
- `ulimits`: Ulimit settings for the container.
- `labels`: Docker labels.
- `pull_policy`: Image pull policy ("missing" by default).
- `healthcheck`: Docker healthcheck configuration.
- `extra_hosts`: Extra hosts to add to container.


------

> **The following sections describe future and alternative design options for the ModelAdapter interface.**  
> Included for reference and to inform future development. 


------

> **Note:** The following design is ambitious and would require significant development effort. Due to time constraints, it is not feasible to implement in the current release and may be considered for a future version.

## ModelAdapter Design (D) - Future Release.

A design that leverages structured model and dataset specifications (`ModelSpec`, `InputOutputSpecs`, `Resource`), enabling rich validation, compatibility checks, and declarative configuration. This approach supports advanced schema-driven validation, resource requirements, and extensibility for complex benchmarking workflows.

### `InputOutputSpecs`


```python
class InputOutputSpecs(BaseModel, extra="forbid"):
    name        : str                                 # Human-readable specification name
    modality    : str                                 # e.g. "rna", "dna", "image"
    semantic    : str                                 # Ontology reference (EDAM, OBI, GO, etc.)
    organism    : str
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

### Model Specifications

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

> Note: For complete details on above specifications, please refer to PR #281

```py
# Useful to validate hardware resource requirement for a model
class Resource(BaseModel, extra="forbid"):
  use_gpu            : bool = False
  gpu_type       : str
  gpu_count      : int = 1                              # Minimum number of GPUs required
  gpu_min_memory     : Optional[int] = None                 # Minimum memory per GPU in MB
  gpu_features       : set[str] = set()                     # e.g., {"flash_attention", "fp8"}
  gpu_mem_mb     : Optional[int] = None                 # Total GPU memory required (legacy, prefer min_mem_mb)
  gpu_compute_capability : Optional[str] = None         # e.g., "7.0", "8.0" (NVIDIA CUDA compute capability)
  multi_gpu      : Optional[bool] = False               # Whether multi-GPU is required
  cpu_cores      : int = 4
  ram_mb         : int = 8_192
  disk_mb        : int = 10_240
  cloud_provider : Optional[Literal["aws", "gcp", "azure", "custom"]] = None
  instance_type  : Optional[str] = None                 # e.g., "n1-standard-8", "m5.large"
  region         : Optional[str] = None                 # e.g., "us-west-2"
```

### Validation Results (Optional)

A common class across cz-benchmarks to report validation results

```py
@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```


```py

class ModelAdapter(ABC):
    """
    Sequence (called by Runner):
      validate → setup → preprocess → predict → postprocess → cleanup
    """

    def __init__(
        self, 
        model_spec: Optional[ModelSpec | str] = None,
        resource_hint: Optional[Resource] = None,
        cfg_file: Optional[str] = None,
        **cfg_overrides,
    ):
        self.model_spec = model_spec
        self.config = Utils.load_cfg(cfg_file, **cfg_overrides) # Add new method in utils


     
    def validate(self, dataset: BaseDataset, Optional[dict[str, Any]]) -> List[ValidationResult]:
            """
            Validates the provided dataset using predefined and custom validation checks.

            Args:
                dataset (BaseDataset): The dataset to be validated.
                options (Optional[dict[str, Any]]): Optional dictionary of additional validation options.

            Returns:
                List[ValidationResult]: A list of validation results for the dataset.

            Notes:
                Base classes can extend this method and call `super().validate()` to leverage predefined validation checks,
                while also adding custom validation logic as needed.
            """

        # Call compatibility checker from PR #281
        validation_messages = CompatibilityChecker.check_compatibility(
            dataset=dataset.dataset_spec, # Add missing property to cz-benchmark BaseDataset class 
            model=self.model_spec,
        )
        return validation_messages
        # Optional call declarative validation (from PR #281) to validate if dataset is as per model requirements

    @abstractmethod
    def setup(self) -> None:                        
        pass

    @abstractmethod
    def preprocess(self, dataset: BaseDataset) -> BaseDataset:
        return dataset

    @abstractmethod
    def predict(
        self,
        batch_size: int | None = None,
        **model_params,
    ) -> Any: ...

    @abstractmethod
    def postprocess(self, prediction_path: str) -> ModelSpec: 
        pass

    def cleanup(self) -> None:                     
        pass

    # Optional
    @abstractmethod
    async def predict_async(...):                   # optional async version
        raise NotImplementedError

    # --- fault tolerance hooks --- #
    @abstractmethod
    def checkpoint(self)->dict[str,Any]: return {}
    @abstractmethod
    def resume(self,state:dict[str,Any]): pass
```

## Hydra/OmegaConf

Add to common util python

```py

# Define env variable to configure or config file or any other mechanism to set default
import os

try:
    import hydra, omegaconf
    _HYDRA = True
except ImportError:
    _HYDRA = False

# Example: override via environment variable
if os.environ.get("USE_HYDRA", "").lower() in {"1", "true", "yes"}:
    _HYDRA = True
elif os.environ.get("USE_HYDRA", "").lower() in {"0", "false", "no"}:
    _HYDRA = False

def load_cfg(path, **overrides): 
    if not path:
        return {}
    if _HYDRA:
        cfg = omegaconf.OmegaConf.load(path)
        return omegaconf.OmegaConf.merge(cfg, overrides)
    with open(path) as fp:
        return yaml.safe_load(fp)
```

Sample code and may be added in a different way. 


### Advantages

- **Strong Typing & Validation:** Leverages Pydantic models for schema validation, ensuring correctness, clarity, and early error detection.
- **Rich Input/Output Specification:** Explicitly models input/output modalities, file formats, encodings, and validation schemas for robust interoperability.
- **Resource Management:** Clearly defines and types resource requirements (CPU, GPU, RAM, etc.) for better scheduling and compatibility checks.
- **Structured Validation Results:** Returns detailed validation results with errors and warnings, improving error handling and reporting.
- **Extensible:** Designed to support advanced adapters and complex benchmarking workflows, making it suitable for extensible and future-proof systems.
- **Provenance & Metadata:** Supports tracking of model origin, authorship, and arbitrary metadata for reproducibility and auditability.
- **Advanced Config Management:** Optional Hydra/OmegaConf integration enables flexible, hierarchical, and override-friendly configuration management.
- **Declarative Validation:** Supports compatibility and optional schema-driven validation, enabling automated and declarative checks. Ability to reuse pre-defined checks.

### Drawbacks

- **Complexity:** More complex to implement and understand, especially for simple or lightweight use cases.
- **Dependency Overhead:** Introduces dependencies on Pydantic and, optionally, Hydra/OmegaConf, which may not be desirable in all environments.

------

# Appendix

## Other designs

## ModelAdapter Design (A)

A minimal, standardized abstract base class for inference adapters, providing essential lifecycle hooks (`validate`, `setup`, `preprocess`, `postprocess`, `run`, `cleanup`) to ensure consistent implementation across different models, while leaving configuration and validation details to model developers.

```py
from typing import Dict, Any, Optional, List

class ModelAdapter(ABC):
    """
    Abstract base class for all inference adapters.

    ModelAdapter provides a standardized interface and common functionality for loading configurations,
    validating them, and setting up resources required for inference tasks. All concrete adapters must
    implement the `run()` method to define their inference logic.

    Key Methods:
       
        - validate():
            Validates the configuration dictionary. This method is intended to be overridden by subclasses
            to enforce required configuration keys or value constraints.
        - setup():
            Optional setup step for initializing resources or performing any pre-inference preparation.
            Subclasses can override this method as needed.
        - preprocess(dataset) -> dataset
        - postprocess(model_output) -> task_input
        - run():
            Abstract method that must be implemented by subclasses. Executes the inference process using the
            provided input and parameters, and returns a dictionary mapping artifact keys to file paths.
    """

    def __init__(self) = None):
        """
        Initialize the ModelAdapter.
        """

    @abstractmethod
    def validate(self):
        """
        Validate the configuration dictionary.

        This method should be overridden by subclasses to enforce required configuration keys or value constraints.
        """
        pass
    @abstractmethod
    def setup(self):
        """
        Optional setup step for initializing resources or performing any pre-inference preparation.

        Subclasses can override this method as needed.
        """
        pass
    @abstractmethod
    def preprocess(self, dataset: Any) -> Any:
        """
        Preprocess the dataset.

        Args:
            dataset (Any): Input dataset.

        Returns:
            Any: Preprocessed dataset.
        """
        return dataset

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """
        Postprocess the model output.

        Args:
            model_output (Any): Model output.

        Returns:
            Any: Postprocessed output.
        """

    @abstractmethod
    def run(
        self, 
        input_path: str, 
        output_path: str,
        batch_size: int | None = None,
        **model_params,
    ) -> Dict[str, str]:
        """
        Run inference. Return a dict mapping artifact keys → file paths (e.g. {"predictions": "/tmp/out.json"}).
        Subclasses must implement this method.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Optional cleanup step to remove temporary files or reset state.
        Subclasses can override this method.
        """
        pass


```

### Drawbacks of Design (A)

- **Lack of Input/Output Schema:** There is no mechanism to specify or validate the types, modalities, or file formats for inputs and outputs. This limits interoperability and makes it difficult to enforce data consistency across adapters.
- **No Resource Management:** The design does not provide a way to declare or check resource requirements such as GPU, RAM, or other hardware needs, making it unsuitable for environments where resource constraints are important.
- **Unstructured Validation:** The `validate` method is only a stub, with no defined return type or structured error reporting. This makes it hard to communicate validation results or handle errors systematically.

-----

## ModelAdapter Design (B)

> Alternate design with explicit arguments and structured configuration, supporting both direct dictionary input and config files, and providing full lifecycle management for model adapters.

```py
import yaml
from typing import Dict, Any, List, Optional, Union

class ModelAdapter(ABC):
    """
    Abstract base class for inference adapters, with full lifecycle management.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None
    ):
        self.config: Dict[str, Any] = self.load_config(config, config_file)
        if not self.config:
            raise ValueError("Configuration must be provided via config or config_file")
        self._validate_config_schema(self.config)
        self.resource: Dict[str, Any] = self.config.get('resource', {})
        self.model_validation: Dict[str, Any] = self.config.get('model_validation', {})
        self.input_type: Optional[str] = self.config.get('input', None)
        self.model_params: Dict[str, Any] = self.config.get('model_params', {})
        self.output_config: Dict[str, Any] = self.config.get('output', {})
        self.inputs: Dict[str, Any] = self.config.get('inputs', {})

    @staticmethod
    def load_config(
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None
    ) -> Dict[str, Any]:
        if config and config_file:
            raise ValueError("Both config and config_file cannot be provided")
        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        return config or {}

    def supported_modalities(self) -> List[str]:
        """Return supported input modalities."""
        return self.model_validation.get('supported_modalities', [])

    def supported_filetypes(self) -> List[str]:
        """Return supported input file types."""
        return self.model_validation.get('supported_filetypes', [])

    def input_schema(self) -> Dict[str, Any]:
        """Return input schema for validation."""
        return self.model_validation.get('input_schema', {})

    @abstractmethod
    def validate(self, inputs: Dict[str, Any]) -> None:
        """
        Validate input data type, file type, and schema.
        Raise exception if validation fails.
        Args:
            inputs: Input data to validate.
        """
        pass

    @abstractmethod
    def setup(self) -> None:
        """
        Prepare resources (e.g. clone repo, create venv).
        Should be idempotent.
        """
        pass

    @abstractmethod
    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """
        Preprocess input data before inference.
        Args:
            inputs: Raw input data.
        Returns:
            Preprocessed data to be passed to inference.
        """
        pass

    @abstractmethod
    def postprocess(self, inference_outputs: Any) -> Any:
        """
        Postprocess inference outputs (e.g. format results, save logs).
        Args:
            inference_outputs: Raw outputs from inference.
        Returns:
            Final results as a dictionary.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Tear down resources (e.g. temp dirs).
        Should be idempotent.
        """
        pass

    @abstractmethod
    def _run(self, preprocessed_inputs: Any) -> Any:
        """
        Core inference logic.
        Args:
            preprocessed_inputs: Data after preprocessing.
        Returns:
            Raw inference outputs.
        """
        pass

    def run(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the full adapter lifecycle.
        Args:
            inputs: Input data for inference.
        Returns:
            Final inference results.
        """
        self.logger.info("=== Starting inference pipeline ===")
        try:
            self.setup()
            self.validate(inputs)
            preprocessed = self.preprocess(inputs)
            raw_outputs = self._run(preprocessed)
            results = self.postprocess(raw_outputs)
            return results
        finally:
            self.cleanup()

```

### Advantages

- **Structured Configuration:** Supports both dictionary and file-based configuration, with schema validation for flexibility and consistency.
- **Lifecycle Management:** Clearly orchestrates all steps in the adapter pipeline (setup, validate, preprocess, run, postprocess, cleanup).
- **Input/Output Validation:** Provides methods to specify and check supported modalities, file types, and input schemas.
- **Separation of Concerns:** Each pipeline step is encapsulated in a dedicated method, improving maintainability.
- **Idempotent Setup/Cleanup:** Encourages robust resource management by making setup and cleanup repeatable and safe.

### Drawbacks

- **Manual Schema Validation:** Relies on runtime checks for configuration and input validation, rather than static typing (e.g., Pydantic models).
- **Lack of Strong Typing:** Configuration and input data are handled as untyped dictionaries, reducing type safety.
- **No Resource/Provenance Classes:** Resource requirements and provenance information are represented as plain dicts, not structured objects.
- **No Structured Validation Results:** Validation errors are raised as exceptions instead of being returned as structured result objects.
- **Limited Validation Features:** Does not support advanced compatibility checks or declarative validation mechanisms.

-----

## **Comparison**

| Feature/Aspect             | Design A | Design B | Design C       | Design D          |
|                                  | -------- | -------- |                 - |                 ---- |
| **Lifecycle Steps**        | Yes      | Yes      | Yes            | Yes               |
| **Config Handling**        | No       | Yes      | Yes (typed)    | Yes (typed, adv.) |
| **Input/Output Schema**    | No       | Partial  | No             | Yes (rich, typed) |
| **Resource Spec**          | No       | Partial  | Yes (typed)    | Yes (typed, rich) |
| **ValidationResult**       | No       | No       | No             | Yes (structured)  |
| **Strong Typing**          | No       | No       | Yes (Pydantic) | Yes (Pydantic)    |
| **Extensibility**          | Medium   | High     | High           | Very High         |
| **Provenance/Metadata**    | No       | No       | No             | Yes               |
| **Declarative Validation** | No       | No       | No             | Yes               |
| **Hydra/OmegaConf**        | No       | No       | No             | Optional          |
| **Complexity**             | Low      | Medium   | Medium         | High              |
| **Ease of Use**            | High     | Medium   | Medium         | Medium/Low        |
| **Production-Ready**       | No       | Partial  | Yes            | Yes               |


