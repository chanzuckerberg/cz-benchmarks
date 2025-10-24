import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator
from czbenchmarks.datasets.types import Organism
from czbenchmarks.cli.utils import CLIError, handle_cli_error


logger = logging.getLogger(__name__)


def _normalize_label_param(name: str, value: Any) -> Any:
    """Normalize common label-like parameters to AnnData references.

    Converts simple column names like 'cell_type' to '@obs:cell_type' format.
    Handles both single values and lists of values.

    Args:
        name: Parameter name
        value: Parameter value to normalize

    Returns:
        Normalized value with proper AnnData reference format
    """
    # Consider using regex or faster data structure for highly performant code
    label_like = {"labels", "input_labels", "batch_labels", "sample_ids"}

    if name not in label_like or not value:
        return value

    # Handle single string value
    if isinstance(value, str):
        if not value.startswith("@"):
            return f"@obs:{value}"
        return value

    # Handle list/tuple of values
    if isinstance(value, (list, tuple)):
        normalized = []
        for v in value:
            if isinstance(v, str) and not v.startswith("@"):
                normalized.append(f"@obs:{v}")
            else:
                normalized.append(v)
        return normalized

    return value


def _to_organism_enum(s: str) -> Organism:
    """Convert a string to an Organism enum, case-insensitive.

    Supports enum names (e.g. "HUMAN"), species names (e.g. "homo_sapiens"),
    gene prefix (e.g. "ENSG"), and common names like "human" or "mouse".

    Args:
        s: String to convert to Organism enum

    Returns:
        Organism enum value

    Raises:
        ValueError: If the string cannot be converted to an Organism
    """
    user_organism = s.strip().lower()
    for organism in Organism:
        logger.debug(
            f"Resolving organism from: {user_organism}, to: {organism._name_.lower()}, "
            f"{organism.value[0].lower()}, {organism.value[1].lower()}"
        )
        # Match enum name (e.g. "HUMAN")
        if user_organism == organism._name_.lower():
            return organism
        # Match species name (e.g. "homo_sapiens")
        if user_organism == organism.value[0].lower():
            return organism
        # Match gene prefix (e.g. "ENSG")
        if organism.value[1] and user_organism == organism.value[1].lower():
            return organism

    valid_names = ", ".join([org._name_ for org in Organism])
    logger.error(
        f"Cannot convert '{s}' to Organism enum. Valid values are: {valid_names}"
    )
    raise ValueError(
        f"Cannot convert '{s}' to Organism enum. Valid values are: {valid_names}"
    )


def _align_labels_to_organisms(
    labels: List[str], organisms: List[str], default_label: str = "cell_type"
) -> Tuple[List[str], List[Organism]]:
    """Align and process labels and organisms for cross-species tasks.

    - Ensures each organism has a corresponding label
    - Converts label names to AnnData reference format if needed
    - Pads labels with default if fewer labels than organisms
    - Parses organism strings into Organism enums

    Args:
        labels: List of label strings (may be column names or references)
        organisms: List of organism strings (e.g., "homo_sapiens", "HUMAN", etc.)
        default_label: Default label to use if not enough labels provided

    Returns:
        Tuple of (processed_labels, organisms_list)
            processed_labels: List of label references, one per organism
            organisms_list: List of Organism enums
    """
    num_organisms = len(organisms)
    labels = list(labels) if labels else []

    # Pad labels if fewer than organisms
    if len(labels) < num_organisms:
        labels += [default_label] * (num_organisms - len(labels))
    # Truncate if more labels than organisms
    if len(labels) > num_organisms:
        labels = labels[:num_organisms]

    processed_labels = []
    for idx, label in enumerate(labels):
        if isinstance(label, str) and label.startswith("@"):
            processed_labels.append(label)
        else:
            # For multi-dataset, use indexed references
            processed_labels.append(f"@{idx}:obs:{label}")

    organisms_list = []
    for organism_str in organisms:
        # Remove any suffix after colon if present
        org_name = organism_str.split(":", 1)[0]
        org_enum = _to_organism_enum(org_name)
        organisms_list.append(org_enum)

    return processed_labels, organisms_list


def _user_dataset_key(path: Path) -> str:
    """
    Generate a robust, unique key for user datasets.

    Uses file metadata (path, size, modification time) to create a content-aware
    hash that avoids cache collisions while being fast to compute.

    Args:
        path: Path to the user dataset file

    Returns:
        str: A unique dataset key in format 'user_dataset_{sanitized_name}_{hash}'
    """
    try:
        resolved_path = path.resolve()
        stat_info = resolved_path.stat()

        payload = (
            f"{resolved_path}|{stat_info.st_size}|{int(stat_info.st_mtime)}".encode()
        )
        content_hash = hashlib.sha256(payload).hexdigest()[:12]

    except Exception:
        content_hash = hashlib.sha256(str(path).encode()).hexdigest()[:12]

    sanitized_name = "".join(c if c.isalnum() else "_" for c in path.stem)

    return f"user_dataset_{sanitized_name}_{content_hash}"


class UserDatasetSpec(BaseModel):
    """
    Pydantic model for user-provided dataset configuration.

    Validates and manages user dataset specifications including the dataset
    class, organism type, and file path. Automatically expands user paths
    and validates file existence.

    Attributes:
        dataset_class (str): Fully qualified class name for the dataset.
        organism (str): Organism type (e.g., "HUMAN", "MOUSE").
        path (Path): Resolved path to the dataset file.
    """

    dataset_class: str
    organism: str
    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def validate_path_exists_and_expand(cls, v):
        p = Path(v).expanduser()
        if not p.exists():
            handle_cli_error(CLIError(f"User dataset file not found: {p}"))
        return p

    model_config = {"extra": "forbid", "protected_namespaces": ()}


class BenchmarkRunSpec(BaseModel):
    """
    Complete specification for running a benchmark evaluation.

    Defines all parameters needed to execute a benchmark including model
    selection, dataset configuration, task specification, and baseline
    options. Supports both VCP models and precomputed cell representations,
    as well as both czbenchmarks datasets and user-provided datasets.

    Attributes:
        czb_dataset_key (Optional[str]): czbenchmarks dataset identifier.
        user_dataset (Optional[UserDatasetSpec]): User-provided dataset config.
        task_key (Optional[str]): Benchmark task identifier.
        cell_representation (Optional[Path]): Path to precomputed embeddings.
        run_baseline (bool): Whether to compute baseline metrics.
        baseline_args (Optional[Dict[str, Any]]): Baseline computation parameters.
    """

    czb_dataset_keys: List[str] = []
    user_datasets: List[UserDatasetSpec] = []
    task_key: str
    # str support both paths or AnnData refs from dataset
    cell_representations: List[str] = []
    run_baseline: bool = False
    baseline_args: Optional[Dict[str, Any]] = None
    task_inputs: Optional[Dict[str, Any]] = None

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "protected_namespaces": (),
    }

    @classmethod
    def from_cli_args(cls, args: Dict) -> "BenchmarkRunSpec":
        """
        Create a BenchmarkRunSpec from CLI arguments.

        Parses and validates CLI arguments to create a complete benchmark
        specification. Handles JSON parsing for user datasets and baseline
        arguments, and resolves benchmark keys to individual components.

        Now includes comprehensive validation using TaskRegistry:
        - Label normalization (converts plain names to @obs:name)
        - Organism resolution and validation
        - Multi-dataset label alignment with indexed references
        - Parameter validation before expensive operations

        Args:
            args (Dict): Dictionary of CLI arguments from click.

        Returns:
            BenchmarkRunSpec: Validated benchmark specification.

        Raises:
            CLIError: If required arguments are missing, JSON parsing fails,
                    benchmark key resolution fails, or validation fails.
        """
        from czbenchmarks.tasks.task import TASK_REGISTRY

        cell_reps = list(args.get("cell_representation_path", ()) or [])
        czb_dataset_keys = list(args.get("dataset", ()) or [])
        user_dataset_specs = list(args.get("user_dataset", ()) or [])
        task_key = args.get("task_key")  # add from subcommand

        # Validate task_key before any expensive operations
        if not task_key:
            handle_cli_error(
                CLIError("Task key is required. Use --help to see available tasks.")
            )

        try:
            task_info = TASK_REGISTRY.get_task_info(task_key)
        except ValueError as e:
            handle_cli_error(CLIError(f"Invalid task: {e}"))

        spec_data: Dict[str, Any] = {
            "czb_dataset_keys": czb_dataset_keys,
            "cell_representations": cell_reps,
            "task_key": task_key,
            "run_baseline": args.get("run_baseline", False),
        }

        if user_dataset_specs:
            try:
                parsed: List[UserDatasetSpec] = []
                for item in user_dataset_specs:
                    parsed.append(UserDatasetSpec(**json.loads(item)))
                spec_data["user_datasets"] = parsed
            except Exception as e:
                handle_cli_error(CLIError(f"Invalid user dataset: {e}"))

        # Extract and normalize task parameters from CLI
        task_kwargs = {}
        baseline_kwargs = {}

        # Extract task parameters
        for param_name in task_info.task_params:
            if param_name in args and args[param_name] is not None:
                val = args[param_name]
                # Skip empty tuples from multiple params
                if isinstance(val, tuple):
                    if not val:
                        continue
                    # Convert tuple to list for consistency
                    val = list(val)
                # Normalize label-like parameters
                task_kwargs[param_name] = _normalize_label_param(param_name, val)

        # Extract baseline parameters
        for param_name in task_info.baseline_params:
            baseline_key = f"baseline_{param_name}"
            if baseline_key in args and args[baseline_key] is not None:
                val = args[baseline_key]
                if isinstance(val, tuple):
                    if not val:
                        continue
                    val = list(val)
                baseline_kwargs[param_name] = val

        # Special handling for multi-dataset tasks with organisms
        if task_info.requires_multiple_datasets:
            # Handle organism_list or organisms parameter
            org_key = None
            for possible_key in ["organism_list", "organisms"]:
                if possible_key in task_kwargs:
                    org_key = possible_key
                    break

            if org_key:
                organisms = task_kwargs[org_key]
                if not isinstance(organisms, list):
                    organisms = [organisms]

                # Get labels if present
                label_key = None
                for key in ["labels", "input_labels"]:
                    if key in task_kwargs:
                        label_key = key
                        break

                labels = task_kwargs.get(label_key, []) if label_key else []
                if not isinstance(labels, list):
                    labels = [labels]

                try:
                    aligned_labels, organism_enums = _align_labels_to_organisms(
                        labels, organisms
                    )
                    if label_key:
                        task_kwargs[label_key] = aligned_labels
                    task_kwargs[org_key] = organism_enums
                except ValueError as e:
                    handle_cli_error(CLIError(f"Organism resolution failed: {e}"))

        # Merge with any existing task_inputs/baseline_args from JSON
        if args.get("task_inputs"):
            try:
                json_task_inputs = json.loads(args["task_inputs"])
                task_kwargs = {**json_task_inputs, **task_kwargs}
            except Exception as e:
                handle_cli_error(CLIError(f"Invalid task inputs: {e}"))

        if args.get("baseline_args"):
            try:
                json_baseline_args = json.loads(args["baseline_args"])
                baseline_kwargs = {**json_baseline_args, **baseline_kwargs}
            except Exception as e:
                handle_cli_error(CLIError(f"Invalid baseline args: {e}"))

        # Store task inputs (validation will happen later in run_task after reference resolution)
        # We skip Pydantic validation here because AnnData references like '@obs'
        # need to be resolved first before they can be validated as actual DataFrames/arrays
        if task_kwargs:
            # Basic validation: check for required parameters
            for param_name, param_info in task_info.task_params.items():
                if param_info.required and param_name not in task_kwargs:
                    handle_cli_error(
                        CLIError(
                            f"Missing required parameter for task '{task_key}': {param_name}"
                        )
                    )
            spec_data["task_inputs"] = task_kwargs

        # Store baseline args (validation will happen later)
        if baseline_kwargs:
            # Basic validation: check for required parameters
            for param_name, param_info in task_info.baseline_params.items():
                if param_info.required and param_name not in baseline_kwargs:
                    handle_cli_error(
                        CLIError(
                            f"Missing required parameter for baseline of task '{task_key}': {param_name}"
                        )
                    )
            spec_data["baseline_args"] = baseline_kwargs
            spec_data["run_baseline"] = True

        spec = cls(**spec_data)

        # Final validation
        if not (
            (spec.cell_representations)
            and (spec.czb_dataset_keys or spec.user_datasets)
            and spec.task_key
        ):
            handle_cli_error(
                CLIError(
                    "Missing required arguments: model/cell_representation, dataset/user-dataset. Use --help for details."
                )
            )

        logger.info(
            f"Selected benchmark run - "
            f"Dataset: {spec.dataset_key}, "
            f"Task: {spec.task_key}"
        )

        return spec

    def _parse_dynamic_params(self, value: Any) -> Any:
        """
        Parse dynamic CLI parameter values for task/baseline params.

        Handles AnnData references, JSON strings, and passthrough values.
        """
        if isinstance(value, str) and value.startswith("@"):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _normalize_cli_param_values(
        self, raw: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Normalize CLI parameter values for task/baseline params.

        Converts stringified JSON and AnnData references to usable Python objects.
        """
        if not raw:
            return {}
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, tuple):
                continue
            else:
                out[k] = self._parse_dynamic_params(v)
        return out

    @property
    def key(self) -> str:
        """
        Generate a unique key for this benchmark run.

        Creates a composite key from model, dataset, and task for caching
        and identification purposes.

        Returns:
            str: Formatted key as "model-dataset-task".
        """

        return f"{self.dataset_key}-{self.task_key}"

    @property
    def dataset_keys(self) -> List[str]:
        """
        Get the dataset key for this benchmark run.

        Returns either the czbenchmarks dataset key or a generated key
        for user datasets based on the filename.

        Returns:
            str: Dataset key for caching and identification.

        Note:
            For user datasets, generates a sanitized key from the filename.
            This may cause cache collisions for files with identical names.
        """

        keys: List[str] = []
        keys.extend(self.czb_dataset_keys)
        for ud in self.user_datasets:
            keys.append(_user_dataset_key(Path(ud.path)))
        return keys

    @property
    def dataset_key(self) -> str:
        """
        Get the dataset key for this benchmark run.

        Returns either the czbenchmarks dataset key or a generated key
        for user datasets based on the filename.

        Returns:
            str: Dataset key for caching and identification.

        Note:
            For user datasets, generates a sanitized key from the filename.
            This may cause cache collisions for files with identical names.
        """

        assert self.user_dataset or self.czb_dataset_key
        if self.user_dataset:
            return _user_dataset_key(Path(self.user_dataset.path))
        return self.czb_dataset_key

    @property
    def czb_dataset_key(self) -> Optional[str]:
        """Return the first czb dataset key if available."""
        return self.czb_dataset_keys[0] if self.czb_dataset_keys else None

    @property
    def cell_representation(self) -> Optional[str]:
        """Return the first cell representation if available."""
        return self.cell_representations[0] if self.cell_representations else None

    @property
    def user_dataset(self) -> Optional[UserDatasetSpec]:
        return self.user_datasets[0] if self.user_datasets else None
