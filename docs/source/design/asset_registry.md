# Asset Manager Design 

The Asset Manager provides a unified, extensible interface for managing all assets involved in computational biology benchmarking workflows. It is responsible for the validation, storage, retrieval, and tracking of diverse artifact types—including datasets, model files, configuration files, and metrics—used in AI/ML benchmarking pipelines.

## Class Definitions

### `BaseAssetManager`

Central class managing assets leveraging storage, caching, and registry.

Responsible for common operations:

* `get()`: Retrieve and optionally cache artifacts.
* `put()`: Validate, store, and register artifacts with metadata.
* `stream()`: Efficient streaming for large files.
* `register_validator()`: Attach custom validators.
* Common metadata management: provenance, versioning, relationships, access control(present in Storage layer).

```python
class BaseAssetManager:
    def get(self, key: str, version: Optional[str] = None) -> pathlib.Path:
        """Retrieve artifact with caching support."""

    def put(self, key: str, local_path: pathlib.Path, metadata: Dict[str, Any], specs: Optional[List[InputOutputSpecs]], **kwargs):
        """Store and validate artifact; register metadata."""

        # InputOutputSpecs validation
        # Optional - Declarative validation (YAML schema) to validate Data QUality
        # Register in registry with extended metadata as below or defined using InputOutputSpecs

        # Example:
        # asset_metadata = {
        #     **metadata,
        #     "version": version,
        #     "provenance": provenance, 
        #     "relationships": relationships,
        #     "access_control": access_control,
        #     "specs": [s.dict() for s in specs] if specs else None,
        #     "declarative_schema": declarative_schema_path,
        # }

        # Upload the asset to storage

    def stream(self, key: str, version: Optional[str] = None, chunk_size: int = 8192):
        """Stream large artifacts efficiently."""

    def register_validator(self, validator_callable):
        """Attach custom Python-based validators."""

    # Optional - Additional metadata management methods:

    def annotate(self, key: str, annotation: str, user: str, version: Optional[str] = None):
        """Annotate asset for audit and tracking."""

    def set_access_control(self, key: str, access_control: Dict[str, Any], version: Optional[str] = None):
        """Set access control rules for securing the Assets"""

    def get_metadata(self, key: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # If VCP integration, call VCP API to get the metadata

    def find_assets(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # If VCP integration, call VCP API to find the asssets

    def list_versions(self, key: str) -> List[str]:
        # If VCP integration, call VCP API

    # Other methods from VCP API can be integrated

```

### `DatasetAssetManager`

Extends BaseAssetManager and  adds validation and metadata specific to datasets.

**Dataset-specific validations:**

* Check modality (`rna`, `dna`, `protein`, `metabolite`, `clinical`, `environmental`)
* Optional - Declarative schema validation (AnnData, Zarr, H5AD, etc.)
* Metadata integrity checks (checksums, provenance, organism validation).

```python
class DatasetAssetManager(BaseAssetManager):
    def put(self, key: str, local_path: pathlib.Path, metadata: Dict[str, Any], specs: List[InputOutputSpecs], declarative_schema: Optional[str]):
        """Store dataset after modality and schema validation."""
        # Dataset-specific validation using InputOutputSpecs
            # raise ValueError(f"Specific failure related error") # In case of validation failure 
        # Optionally, add more dataset-specific checks here
        # Call base put to upload
        super().put()
```

### `ModelAssetManager`

Manages computational biology model artifacts and metadata.

**Model-specific validations:**

* Validate input/output compatibility using `InputOutputSpecs`.
* Validate compute resource requirements, model versioning, provenance tracking.

```python
class ModelAssetManager(BaseAssetManager):
    def put(self, key: str, local_path: pathlib.Path, metadata: Dict[str, Any], specs: List[InputOutputSpecs], declarative_schema: Optional[str]):
        """Store model artifact after comprehensive spec validation."""
        # Model-specific validation using InputOutputSpecs
        # Optionally, add more model-specific checks here
        # Call base put
```


### `ConfigAssetManager`

Handles pipeline, environment, and benchmark configuration files.

**Config-specific validations:**

* Declarative schema validation for YAML/JSON configs.
* Version and dependency checks between config files.

```python
class ConfigAssetManager(BaseAssetManager):
    def put(self, key: str, local_path: pathlib.Path, metadata: Dict[str, Any], declarative_schema: Optional[str]):
        """Store and validate configuration files."""
```


### Support Metadata, Versioning, and Provenance Tracking

Every asset is tracked with comprehensive metadata including:

* **Provenance**: Author, source, citations, DOI.
* **Versioning**: Semantic versioning (v1.0.0, v1.0.1, etc.).
* **Relationships**: Links between artifacts (e.g., dataset ↔ models ↔ tasks ↔ metrics).
* **Access Control**: Permissions (roles, users, groups).


### Streaming Support (Large Files)

* Efficiently stream large artifacts (datasets, models).
* Supports resumable downloads/uploads for robustness.
* Chunked streaming interface (user-defined chunk sizes).


### Additional Asset Types

* Pipeline execution outputs.
* Logs, reports, visualizations.
* Temporary/intermediate results management.



## Class Usage Example

```python
config = AssetConfig("config.yaml")
storage = S3Storage(config)
cache = LocalCache(config)

datasets_manager = DatasetAssetManager(config, storage, cache)

datasets_manager.put(
    key="human_brain_single_cell",
    local_path=pathlib.Path("human_brain.h5ad"),
    metadata={"organism": "human", "modality": "rna"},
    specs=[InputOutputSpecs(modality="rna", semantic="GO:0008150")],
    declarative_schema="schemas/single_cell_anndata.yaml"
)
```



## References

The validation approach described in this document is based on the Input Output Specifications proposed in [cz-benchmarks (PR #281)](https://github.com/chanzuckerberg/cz-benchmarks/pull/281/files). To decouple this design from those specifications, simply remove references to the related classes from the Input Output Specifications documentation.



# Appendix

## Config variables

> **Note:** Ideally, these configuration variables should be directly used by the Storage and Caching components. This ensures that storage backends (like S3, Local, HTTP) and cache implementations (such as LocalCache, RedisCache) can be easily configured and swapped without changing application code. Centralizing these settings improves maintainability and flexibility.

```env
# Configuration file path. Either config.yaml or this .env file
CONFIG_PATH=config.yaml              # Path or URL to config file

# Caching configuration
CACHE_TYPE=LocalCache                # Options: LocalCache, RedisCache, Memcached, etc.
CACHE_ENABLED=true                   # Enable or disable caching (true/false)
CACHE_TTL=3600                       # Cache time-to-live in seconds (optional)

# Storage configuration
STORAGE_TYPE=S3                      # Options: Local, S3, HTTP, HTTPSS, HuggingFace

# S3 Storage settings
S3_BUCKET=my-bucket
S3_PREFIX=my-prefix
S3_REGION=us-west-2
S3_ENDPOINT_URL=                     # Optional: custom S3 endpoint (VAST or other S3 compatible storage)
S3_USE_SSL=true                      # Use SSL for S3 connections (true/false)
AWS_ACCESS_KEY_ID=                   # AWS credentials (if not using IAM roles)
AWS_SECRET_ACCESS_KEY=

# HTTP/HTTPS Storage settings
HTTP_STORAGE_BASE_URL=https://my-storage.example.com
HTTP_STORAGE_AUTH_TYPE=Bearer        # Options: None, Basic, Bearer, APIKey
HTTP_STORAGE_AUTH_TOKEN=             # Token or API key for authentication
HTTP_STORAGE_USERNAME=               # For Basic auth
HTTP_STORAGE_PASSWORD=               # For Basic auth
HTTP_STORAGE_VERIFY_SSL=true         # Verify SSL certificates (true/false)
HTTP_STORAGE_HEADERS=                # Optional: JSON string of extra headers

# Database Options

# Database configuration
DB_TYPE=PostgreSQL                   # Options: PostgreSQL, MySQL, SQLite, MongoDB, etc.
DB_HOST=localhost                    # Database host
DB_PORT=5432                         # Database port
DB_NAME=asset_registry               # Database name
DB_USER=                             # Database username
DB_PASSWORD=                         # Database password
DB_SSL_MODE=prefer                   # SSL mode (disable, allow, prefer, require, verify-ca, verify-full)


# Hugging Face or other remote sources (optional)
HF_TOKEN=                            # Hugging Face access token

# Additional security options
ALLOW_INSECURE_STORAGE=false         # Prevent use of insecure (non-SSL) endpoints

```


## Storage and Cache Methods

* **Storage implementations** (`LocalStorage`, `S3Storage`, `HTTPStorage`) may implement as desired:

  * `download(key: str) -> pathlib.Path`
  * `upload(key: str, local_path: pathlib.Path) -> str` (returns remote path or URL)
  * `stream(key: str, chunk_size: int)` (yields data chunks)
  * **Security**: All access control logic should be enforced both at the registry and storage layers.

* **Cache implementations** (`LocalCache`, `RedisCache`, `S3Cache`) may implement as desired:

  * `get(key: str) -> Optional[pathlib.Path]`
  * `set(key: str, value: pathlib.Path, ttl: int)`




### AssetConfig

Manages and provides configurations for storage, caching, and registry operations.

```python
import os
from dotenv import load_dotenv
import yaml

class AssetConfig: # OR Storage Config
    """
    Loads configuration parameters for the AssetManager from environment variables or YAML configuration.
    """
    def __init__(self, config_path=None):
        load_dotenv()
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                config_yaml = yaml.safe_load(file)
                self.CACHE_TYPE = config_yaml.get('CACHE_TYPE', os.getenv("CACHE_TYPE", "LocalCache"))
                self.CACHE_ENABLED = config_yaml.get('CACHE_ENABLED', os.getenv("CACHE_ENABLED", "true")).lower() == "true"
                self.STORAGE_TYPE = config_yaml.get('STORAGE_TYPE', os.getenv("STORAGE_TYPE", "Local"))
                self.DB_CONFIG = config_yaml.get('DB_CONFIG', {})
        else:
            self.CACHE_TYPE = os.getenv("CACHE_TYPE", "LocalCache")
            self.CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
            self.STORAGE_TYPE = os.getenv("STORAGE_TYPE", "Local")
            self.DB_CONFIG = {
                "DB_TYPE": os.getenv("DB_TYPE", "SQLite"),
                "DB_PATH": os.getenv("DB_PATH", "assets.db")
            }
```

### AssetRegistry

Stores and retrieves metadata about assets. Can be implemented using databases like PostgreSQL, SQLite, or others.

```python
class AssetRegistry: #OR Storage Registry.
    """
    Manages the metadata of registered assets, including storage paths and associated details.
    """

    def __init__(self, config: AssetConfig):
        """
        Initialize the registry database connection.

        Args:
            config (AssetConfig): Configuration object containing database connection details.
        """


    def _initialize_db(self):
        """Initialize databse etc"""


    def register(self, key: str, path: str, metadata: dict):
        """
        Registers a new asset with metadata into the registry.

        Args:
            key (str): Unique identifier for the asset.
            path (str): Storage path or URL to the asset.
            metadata (dict): Additional metadata for the asset.
        """


    def get(self, key: str):
        """
        Retrieves metadata for a registered asset.

        Args:
            key (str): Unique identifier for the asset.

        Returns:
            dict: Asset details or None if not found.
        """
 
```


### Asset Manager


```py
from config import AssetConfig
from storage import Storage
from cache import Cache
from registry import AssetRegistry
from typing import Optional, Dict, Any, List
import pathlib
import logging
import yaml

# Import InputOutputSpecs and validation classes
from input_output_specifications import (
    InputOutputSpecs, DatasetSpec, ModelSpec, TaskSpec, MetricSpec,
    CompatibilityChecker, CompatibilityIssue
)

# --- Base Asset Manager ---

class BaseAssetManager:
    """
    Base class for managing computational biology benchmarking assets.
    Provides common logic for storage, caching, registry, and validation.
    """

    def __init__(self, config: AssetConfig, storage: Storage, cache: Optional[Cache] = None):
        self.config = config
        self.storage = storage
        self.cache = cache if config.CACHE_ENABLED else None
        self.registry = AssetRegistry(config)
        self.validators = []  # Python validator callables
        self.declarative_validators = {}  # {spec_id: validator instance}

    def get(self, key: str, version: Optional[str] = None) -> pathlib.Path:
        cache_key = f"{key}:{version}" if version else key
        if self.cache:
            asset = self.cache.get(cache_key)
            if asset:
                logging.info(f"Cache hit for asset '{cache_key}'")
                return asset
        asset_path = self.storage.download(key, version=version)
        if self.cache:
            self.cache.set(cache_key, asset_path, ttl=self.config.CACHE_TTL)
        return asset_path

    def put(
        self,
        key: str,
        local_path: pathlib.Path,
        metadata: Dict[str, Any],
        specs: Optional[List[InputOutputSpecs]] = None,
        declarative_schema_path: Optional[str] = None,
        relationships: Optional[Dict[str, List[str]]] = None,
        version: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        access_control: Optional[Dict[str, Any]] = None,
    ):
        """
        Store an asset, validate with InputOutputSpecs and declarative schema, and register.
        """
        # 1. InputOutputSpecs validation (Pythonic, fail-fast)
        if specs:
            for validator in self.validators:
                validator(local_path, specs, metadata)

        # 2. Declarative validation (YAML schema)
        if declarative_schema_path:
            with open(declarative_schema_path, "r") as f:
                schema = yaml.safe_load(f)
            validator = self._get_declarative_validator(schema)
            validator.validate_asset(local_path, schema)

        # 3. Upload to storage
        remote_path = self.storage.upload(key, local_path, version=version)

        # 4. Register in registry with extended metadata
        asset_metadata = {
            **metadata,
            "version": version,
            "provenance": provenance,
            "relationships": relationships,
            "access_control": access_control,
            "specs": [s.dict() for s in specs] if specs else None,
            "declarative_schema": declarative_schema_path,
        }
        self.registry.register(key, remote_path, asset_metadata)

    def _get_declarative_validator(self, schema: dict):
        from input_output_specifications import BaseValidator
        return BaseValidator(schema)

    def get_metadata(self, key: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.registry.get(key, version=version)

    def find_assets(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.registry.query(query)

    def list_versions(self, key: str) -> List[str]:
        return self.registry.list_versions(key)

    def set_access_control(self, key: str, access_control: Dict[str, Any], version: Optional[str] = None):
        self.registry.update_access_control(key, access_control, version=version)

    def annotate(self, key: str, annotation: str, user: str, version: Optional[str] = None):
        self.registry.add_annotation(key, annotation, user, version=version)

    def register_validator(self, validator_callable):
        self.validators.append(validator_callable)

    def stream(self, key: str, version: Optional[str] = None, chunk_size: int = 8192):
        yield from self.storage.stream(key, version=version, chunk_size=chunk_size)

# --- Dataset Asset Manager ---

class DatasetsAssetManager(BaseAssetManager):
    """
    Asset manager for computational biology datasets.
    Adds dataset-specific validation and metadata handling.
    """

    def put(
        self,
        key: str,
        local_path: pathlib.Path,
        metadata: Dict[str, Any],
        specs: Optional[List[InputOutputSpecs]] = None,
        declarative_schema_path: Optional[str] = None,
        relationships: Optional[Dict[str, List[str]]] = None,
        version: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        access_control: Optional[Dict[str, Any]] = None,
    ):
        # Dataset-specific validation using InputOutputSpecs
        if specs:
            for spec in specs:
                if spec.modality not in ["rna", "dna", "protein", "metabolite", "clinical", "environmental"]:
                    raise ValueError(f"Unsupported dataset modality: {spec.modality}")
        # Optionally, add more dataset-specific checks here

        # Call base put
        super().put(
            key, local_path, metadata, specs, declarative_schema_path,
            relationships, version, provenance, access_control
        )

# --- Model Asset Manager ---

class ModelsAssetManager(BaseAssetManager):
    """
    Asset manager for computational biology models (weights, artifacts, etc).
    Adds model-specific validation and metadata handling.
    """

    def put(
        self,
        key: str,
        local_path: pathlib.Path,
        metadata: Dict[str, Any],
        specs: Optional[List[InputOutputSpecs]] = None,
        declarative_schema_path: Optional[str] = None,
        relationships: Optional[Dict[str, List[str]]] = None,
        version: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        access_control: Optional[Dict[str, Any]] = None,
    ):
        # Model-specific validation using InputOutputSpecs
        if specs:
            for spec in specs:
                if spec.modality not in ["statistical", "ml", "simulation"]:
                    raise ValueError(f"Unsupported model modality: {spec.modality}")
        # Optionally, add more model-specific checks here

        # Call base put
        super().put(
            key, local_path, metadata, specs, declarative_schema_path,
            relationships, version, provenance, access_control
        )


# --- Config Asset Manager (for config files, pipeline configs, etc.) ---

class ConfigAssetManager(BaseAssetManager):
    """
    Asset manager for configuration files and pipeline configs.
    """

    def put(
        self,
        key: str,
        local_path: pathlib.Path,
        metadata: Dict[str, Any],
        specs: Optional[List[InputOutputSpecs]] = None,
        declarative_schema_path: Optional[str] = None,
        relationships: Optional[Dict[str, List[str]]] = None,
        version: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        access_control: Optional[Dict[str, Any]] = None,
    ):
        # Config-specific validation can be added here
        super().put(
            key, local_path, metadata, specs, declarative_schema_path,
            relationships, version, provenance, access_control
        )

class OutputAssetManager(BaseAssetManager):
    """ Manage outputs produced at various stages of the benchmarking workflow """

```

> **Note:** The code examples provided are for conceptual illustration and design purposes only and ocde have not been validated or tested.

