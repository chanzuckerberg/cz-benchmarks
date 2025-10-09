from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union, Annotated
from pydantic import Field

import anndata as ad
import scipy.sparse as sp
from pydantic import BaseModel, ValidationError
from pydantic.fields import PydanticUndefined
from typing import get_args
from ..constants import RANDOM_SEED
from ..metrics.types import MetricResult
from .types import CellRepresentation
from .utils import run_standard_scrna_workflow


class BaselineInput(BaseModel):
    """Base class for baseline inputs."""

    model_config = {"arbitrary_types_allowed": True}


class PCABaselineInput(BaselineInput):
    """Input for the standard PCA baseline workflow."""

    n_top_genes: int = Field(
        3000, description="Number of highly variable genes for PCA baseline."
    )
    n_pcs: int = Field(
        50, description="Number of principal components for PCA baseline."
    )
    obsm_key: str = Field(
        "emb", description="AnnData .obsm key to store the baseline PCA embedding."
    )


class NoBaselineInput(BaselineInput):
    """A model to signify that no baseline is available for a task."""

    pass


class TaskInput(BaseModel):
    """Base class for task inputs."""

    model_config = {"arbitrary_types_allowed": True}


class TaskOutput(BaseModel):
    """Base class for task outputs."""

    model_config = {"arbitrary_types_allowed": True}


class TaskParameter(BaseModel):
    """Schema for a single, discoverable parameter, including help text and list support."""

    name: str
    type: Any
    stringified_type: str
    default: Any = None
    required: bool
    help_text: str
    is_multiple: bool
    model_config = {"arbitrary_types_allowed": True}


class TaskInfo(BaseModel):
    """Schema for all discoverable information about a single benchmark task."""

    name: str
    display_name: str
    description: str
    task_params: Dict[str, TaskParameter]
    baseline_params: Dict[str, TaskParameter]
    requires_multiple_datasets: bool


class TaskRegistry:
    """Production-grade registry for Task subclasses with comprehensive introspection, validation, and CLI support.

    This registry provides:
    - Automatic task discovery and registration
    - Rich parameter introspection for both Pydantic and function-based tasks
    - Multi-dataset task validation
    - CLI-friendly help text generation
    - Unified validation interface for external programs
    """

    def __init__(self):
        self._registry: Dict[str, Type["Task"]] = {}
        self._info: Dict[str, TaskInfo] = {}

    def register_task(self, task_class: type["Task"]) -> None:
        """Register a Task class and cache its metadata for efficient access.

        Args:
            task_class: The Task subclass to register
        """
        if inspect.isabstract(task_class) or not hasattr(task_class, "display_name"):
            return

        key = (
            getattr(task_class, "display_name", task_class.__name__)
            .lower()
            .replace(" ", "_")
        )
        self._registry[key] = task_class
        self._info[key] = self._introspect_task(task_class)

    @staticmethod
    def _stringify_type(annotation: Any) -> str:
        """Return a string representation of a type annotation.

        Args:
            annotation: The type annotation to stringify

        Returns:
            A human-readable string representation of the type
        """
        try:
            return str(annotation).replace("typing.", "")
        except Exception:
            return str(annotation)

    @staticmethod
    def _is_multiple_type(annotation: Any) -> bool:
        """Determine if a type annotation represents a multiple/list type.

        Handles typing.List, list, Annotated[List], etc.

        Args:
            annotation: The type annotation to check

        Returns:
            True if the annotation represents a list-like type
        """
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, List):
            return True
        # Handle Annotated[List[...], ...]
        if origin is Annotated:
            args = getattr(annotation, "__args__", ())
            if args:
                return TaskRegistry._is_multiple_type(args[0])
        return False

    def _introspect_task(self, task_class: type["Task"]) -> TaskInfo:
        """Extract all metadata for a task using a hybrid strategy.

        Supports both Pydantic models and function signature introspection.

        Args:
            task_class: The Task class to introspect

        Returns:
            TaskInfo object containing all discoverable task metadata
        """
        try:
            # Prefer Pydantic model introspection when available
            if hasattr(task_class, "input_model") and hasattr(
                task_class, "baseline_model"
            ):
                task_params = self._introspect_pydantic_model(task_class.input_model)
                baseline_params = self._introspect_pydantic_model(
                    task_class.baseline_model
                )
            else:
                # Fallback to function signature introspection
                task_params = self._introspect_function_signature(
                    task_class._run_task, exclude={"self", "cell_representation"}
                )
                baseline_params = self._introspect_function_signature(
                    task_class.compute_baseline, exclude={"self", "expression_data"}
                )

            # Introspect requires_multiple_datasets from class or instance
            requires_multiple_datasets = getattr(
                task_class, "requires_multiple_datasets", False
            )
            if not requires_multiple_datasets:
                try:
                    instance = task_class()
                    requires_multiple_datasets = getattr(
                        instance, "requires_multiple_datasets", False
                    )
                except Exception:
                    pass

            return TaskInfo(
                name=task_class.__name__,
                display_name=getattr(task_class, "display_name", task_class.__name__),
                description=inspect.getdoc(task_class)
                or f"No description available for {task_class.__name__}",
                task_params=task_params,
                baseline_params=baseline_params,
                requires_multiple_datasets=requires_multiple_datasets,
            )
        except Exception as e:
            task_name = getattr(task_class, "__name__", "UnknownTask")
            print(f"Warning: Task introspection failed for {task_name}: {e}")
            return TaskInfo(
                name=task_name,
                display_name=getattr(task_class, "display_name", task_name),
                description="Task introspection failed.",
                task_params={},
                baseline_params={},
                requires_multiple_datasets=False,
            )

    @staticmethod
    def _introspect_pydantic_model(
        model: Type[BaseModel] | None,
    ) -> Dict[str, TaskParameter]:
        """Extract rich parameter info from a Pydantic model.

        Args:
            model: The Pydantic model to introspect

        Returns:
            Dictionary mapping parameter names to TaskParameter objects
        """
        if not model:
            return {}
        params = {}
        for name, field in model.model_fields.items():
            annotation = field.annotation
            is_multiple = TaskRegistry._is_multiple_type(annotation)
            params[name] = TaskParameter(
                name=name,
                type=annotation,
                stringified_type=str(annotation).replace("typing.", ""),
                default=field.default
                if field.default is not PydanticUndefined
                else None,
                required=field.is_required(),
                help_text=field.description or "No description provided.",
                is_multiple=is_multiple,
            )
        return params

    @staticmethod
    def _introspect_function_signature(
        func: callable, exclude: set
    ) -> Dict[str, TaskParameter]:
        """Extract basic parameter info from a function's signature as a fallback.

        Used for tasks that don't use Pydantic models.

        Args:
            func: The function to introspect
            exclude: Set of parameter names to exclude

        Returns:
            Dictionary mapping parameter names to TaskParameter objects
        """
        params = {}
        try:
            sig = inspect.signature(func)
            for name, param in sig.parameters.items():
                if name in exclude or param.kind in (
                    param.VAR_KEYWORD,
                    param.VAR_POSITIONAL,
                ):
                    continue
                param_type = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else Any
                )
                is_multiple = TaskRegistry._is_multiple_type(param_type)
                params[name] = TaskParameter(
                    name=name,
                    type=param_type,
                    stringified_type=str(param_type).replace("typing.", ""),
                    default=param.default
                    if param.default != inspect.Parameter.empty
                    else None,
                    required=param.default == inspect.Parameter.empty,
                    help_text="Help text unavailable (defined via function signature).",
                    is_multiple=is_multiple,
                )
        except (ValueError, TypeError):
            pass
        return params

    def list_tasks(self) -> List[str]:
        """Return a sorted list of all available task keys.

        Returns:
            List of task keys that can be used to get task info or classes
        """
        return sorted(self._registry.keys())

    def get_task_info(self, task_name: str) -> TaskInfo:
        """Get all introspected information for a given task.

        Args:
            task_name: The task key (lowercase display name with underscores)

        Returns:
            TaskInfo object containing all task metadata

        Raises:
            ValueError: If the task is not found
        """
        if task_name not in self._info:
            raise ValueError(f"Task '{task_name}' not found.")
        return self._info[task_name]

    def get_task_class(self, task_name: str) -> Type["Task"]:
        """Get the Task class for a given task name.

        Args:
            task_name: The task key (lowercase display name with underscores)

        Returns:
            The Task class

        Raises:
            ValueError: If the task is not found
        """
        if task_name not in self._registry:
            raise ValueError(f"Task '{task_name}' not found.")
        return self._registry[task_name]

    def get_task_help(self, task_name: str) -> str:
        """Generate a human-readable summary string of a task's parameters.

        Perfect for CLI help text generation.

        Args:
            task_name: The task key to generate help for

        Returns:
            Formatted help text string with task description and all parameters
        """
        try:
            info = self.get_task_info(task_name)
            lines = [
                f"Task: {info.display_name}",
                f"Description: {info.description}",
                "",
            ]

            if info.requires_multiple_datasets:
                lines.append("Note: This task requires multiple datasets as input.\n")

            if info.task_params:
                lines.append("Task Parameters:")
                for param in info.task_params.values():
                    default_str = (
                        f" (Default: {param.default})" if not param.required else ""
                    )
                    multiple_str = " [multiple]" if param.is_multiple else ""
                    lines.append(
                        f"  --{param.name.replace('_', '-')} : {param.help_text}{default_str}{multiple_str}"
                    )
                lines.append("")

            TaskClass = self.get_task_class(task_name)
            baseline_model = getattr(TaskClass, "baseline_model", None)
            if baseline_model and baseline_model.__name__ == "NoBaselineInput":
                lines.append("Baseline: This task does not support a baseline.")
            elif info.baseline_params:
                lines.append("Baseline Parameters:")
                for param in info.baseline_params.values():
                    default_str = (
                        f" (Default: {param.default})" if not param.required else ""
                    )
                    multiple_str = " [multiple]" if param.is_multiple else ""
                    lines.append(
                        f"  --baseline-{param.name.replace('_', '-')} : {param.help_text}{default_str}{multiple_str}"
                    )
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"Error generating help for task '{task_name}': {e}"

    def _validate_multi_dataset_consistency(
        self, task_name: str, validated_instance: BaseModel, param_source: str = "task"
    ) -> None:
        """Validate consistency of list-type parameters for multi-dataset tasks.

        Ensures all list parameters have the same length (>1) for multi-dataset tasks.

        Args:
            task_name: The task name for error messages
            validated_instance: The validated Pydantic model instance
            param_source: Either "task" or "baseline" for error messages

        Raises:
            ValueError: If list parameters are inconsistent or have invalid lengths
        """
        info = self.get_task_info(task_name)
        if not info.requires_multiple_datasets:
            return

        param_set = info.task_params if param_source == "task" else info.baseline_params
        multi_params = [p for p in param_set.values() if p.is_multiple]

        if not multi_params:
            return

        lengths = {}
        for param in multi_params:
            value = getattr(validated_instance, param.name, None)
            if value is not None:
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"Parameter '{param.name}' must be a list for multi-dataset task '{task_name}'."
                    )
                lengths[param.name] = len(value)

        if not lengths:
            return

        if any(length < 2 for length in lengths.values()):
            raise ValueError(
                f"Multi-dataset task '{task_name}' requires at least 2 values for list parameters. "
                f"Found: {lengths}"
            )

        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"All list parameters for multi-dataset task '{task_name}' must have the same length. "
                f"Found lengths: {lengths}"
            )

    def validate_task_inputs(
        self, task_name: str, params: Dict[str, Any]
    ) -> TaskInput | Dict:
        """Validate and build task input parameters.

        Returns a Pydantic instance if the task uses Pydantic models, otherwise a dict.
        Performs comprehensive validation including multi-dataset consistency checks.

        Args:
            task_name: The task key
            params: Dictionary of parameter values

        Returns:
            Validated TaskInput instance or dict

        Raises:
            ValueError: If validation fails
        """
        TaskClass = self.get_task_class(task_name)
        InputModel = getattr(TaskClass, "input_model", None)
        info = self.get_task_info(task_name)

        # Validate with Pydantic if possible
        if InputModel:
            try:
                validated = InputModel(**params)
            except ValidationError as e:
                raise ValueError(
                    f"Invalid parameters for task '{task_name}':\n{e}"
                ) from e

            # Multi-dataset consistency validation
            self._validate_multi_dataset_consistency(
                task_name, validated, param_source="task"
            )
            return validated

        # Fallback: manual validation for non-Pydantic tasks
        for name, p_info in info.task_params.items():
            if p_info.required and name not in params:
                raise ValueError(
                    f"Missing required parameter for task '{task_name}': {name}"
                )

        # Multi-dataset validation for dict-based params
        if info.requires_multiple_datasets:
            multi_params = [p for p in info.task_params.values() if p.is_multiple]
            lengths = {}
            for p in multi_params:
                value = params.get(p.name, None)
                if value is not None:
                    if not isinstance(value, (list, tuple)):
                        raise ValueError(
                            f"Parameter '{p.name}' for task '{task_name}' must be a list when multiple datasets are required."
                        )
                    lengths[p.name] = len(value)

            if lengths:
                if any(length < 2 for length in lengths.values()):
                    raise ValueError(
                        f"All list-type parameters for task '{task_name}' must have more than one value. "
                        f"Found: {lengths}"
                    )
                if len(set(lengths.values())) > 1:
                    raise ValueError(
                        f"All list-type parameters for task '{task_name}' must have the same length. "
                        f"Found lengths: {lengths}"
                    )

        return params

    def validate_baseline_inputs(
        self, task_name: str, params: Dict[str, Any]
    ) -> BaselineInput | Dict:
        """Validate and build baseline input parameters.

        Returns a Pydantic instance if the task uses Pydantic models, otherwise a dict.
        Performs comprehensive validation including multi-dataset consistency checks.

        Args:
            task_name: The task key
            params: Dictionary of parameter values

        Returns:
            Validated BaselineInput instance or dict

        Raises:
            ValueError: If validation fails
        """
        TaskClass = self.get_task_class(task_name)
        BaselineModel = getattr(TaskClass, "baseline_model", None)
        info = self.get_task_info(task_name)

        if BaselineModel:
            try:
                validated = BaselineModel(**params)
            except ValidationError as e:
                raise ValueError(
                    f"Invalid parameters for '{task_name}' baseline:\n{e}"
                ) from e

            # Multi-dataset consistency validation for baseline
            self._validate_multi_dataset_consistency(
                task_name, validated, param_source="baseline"
            )
            return validated

        # Fallback: manual validation for non-Pydantic baselines
        for name, p_info in info.baseline_params.items():
            if p_info.required and name not in params:
                raise ValueError(
                    f"Missing required parameter for '{task_name}' baseline: {name}"
                )

        # Multi-dataset validation for dict-based baseline params
        if info.requires_multiple_datasets:
            multi_params = [p for p in info.baseline_params.values() if p.is_multiple]
            lengths = {}
            for p in multi_params:
                value = params.get(p.name, None)
                if value is not None:
                    if not isinstance(value, (list, tuple)):
                        raise ValueError(
                            f"Baseline parameter '{p.name}' for task '{task_name}' must be a list when multiple datasets are required."
                        )
                    lengths[p.name] = len(value)

            if lengths:
                if any(length < 2 for length in lengths.values()):
                    raise ValueError(
                        f"All list-type baseline parameters for task '{task_name}' must have more than one value. "
                        f"Found: {lengths}"
                    )
                if len(set(lengths.values())) > 1:
                    raise ValueError(
                        f"All list-type baseline parameters for task '{task_name}' must have the same length. "
                        f"Found lengths: {lengths}"
                    )

        return params

    def validate_and_build_inputs(
        self, task_name: str, model_type: str, params: Dict[str, Any]
    ) -> BaseModel | Dict:
        """Unified method to validate and build either task or baseline inputs.

        This is a convenience method that routes to the appropriate validation method.
        Useful for external programs that want a single interface.

        Args:
            task_name: The task key
            model_type: Either "input_model" for task inputs or "baseline_model" for baseline inputs
            params: Dictionary of parameter values

        Returns:
            Validated model instance or dict

        Raises:
            ValueError: If validation fails or model_type is invalid
        """
        if model_type == "input_model":
            return self.validate_task_inputs(task_name, params)
        elif model_type == "baseline_model":
            return self.validate_baseline_inputs(task_name, params)
        else:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Must be 'input_model' or 'baseline_model'."
            )


# Global singleton instance, ready for import by other modules.
TASK_REGISTRY = TaskRegistry()


class Task(ABC):
    """Abstract base class for all benchmark tasks.

    Defines the interface that all tasks must implement. Tasks are responsible for:
    1. Declaring their required input/output data types
    2. Running task-specific computations
    3. Computing evaluation metrics

    Tasks should store any intermediate results as instance variables
    to be used in metric computation.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    input_model: Type[TaskInput]
    baseline_model: Type[BaselineInput]  # Add baseline_model attribute

    def __init__(
        self,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        self.random_seed = random_seed
        # FIXME should this be changed to requires_multiple_embeddings?
        self.requires_multiple_datasets = False

    def __init_subclass__(cls, **kwargs):
        """Automatically register task subclasses when they are defined."""
        super().__init_subclass__(**kwargs)
        TASK_REGISTRY.register_task(cls)

    @abstractmethod
    def _run_task(
        self, cell_representation: CellRepresentation, task_input: TaskInput
    ) -> TaskOutput:
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
        Returns:
            TaskOutput: Pydantic model with output data for the task
        """

    @abstractmethod
    def _compute_metrics(
        self, task_input: TaskInput, task_output: TaskOutput
    ) -> List[MetricResult]:
        """Compute evaluation metrics for the task.

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

    def _run_task_for_dataset(
        self,
        cell_representation: CellRepresentation,
        task_input: TaskInput,
    ) -> List[MetricResult]:
        """Run task for a dataset or list of datasets and compute metrics.

        This method runs the task implementation and computes the corresponding metrics.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
        Returns:
            List of MetricResult objects

        """

        task_output = self._run_task(cell_representation, task_input)
        metrics = self._compute_metrics(task_input, task_output)
        return metrics

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        baseline_input: PCABaselineInput = None,
    ) -> CellRepresentation:
        """Set a baseline embedding using PCA on gene expression data."""
        if baseline_input is None:
            baseline_input = PCABaselineInput()

        # Convert sparse matrix to dense if needed for JAX compatibility
        if sp.issparse(expression_data):
            expression_data = expression_data.toarray()

        # Create the AnnData object
        adata = ad.AnnData(X=expression_data)

        # Run the standard preprocessing workflow
        embedding_baseline = run_standard_scrna_workflow(
            adata,
            n_top_genes=baseline_input.n_top_genes,
            n_pcs=baseline_input.n_pcs,
            obsm_key=baseline_input.obsm_key,
            random_state=self.random_seed,
        )
        return embedding_baseline

    def run(
        self,
        cell_representation: Union[CellRepresentation, List[CellRepresentation]],
        task_input: TaskInput,
    ) -> List[MetricResult]:
        """Run the task on input data and compute metrics.

        Args:
            cell_representation: gene expression data or embedding to use for the task
            task_input: Pydantic model with inputs for the task

        Returns:
            For single embedding: A one-element list containing a single metric result for the task
            For multiple embeddings: List of metric results for each task, one per dataset

        Raises:
            ValueError: If input does not match multiple embedding requirement
        """

        # Check if task requires embeddings from multiple datasets
        if self.requires_multiple_datasets:
            error_message = "This task requires a list of cell representations"
            if not isinstance(cell_representation, List):
                raise ValueError(error_message)
            if not all(
                isinstance(emb, get_args(CellRepresentation))
                for emb in cell_representation
            ):
                raise ValueError(error_message)
            if len(cell_representation) < 2:
                raise ValueError(f"{error_message} but only one was provided")
        else:
            if not isinstance(cell_representation, get_args(CellRepresentation)):
                raise ValueError(
                    "This task requires a single cell representation"
                )

        return self._run_task_for_dataset(
            cell_representation,  # type: ignore
            task_input,
        )
