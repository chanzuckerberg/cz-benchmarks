from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

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

    n_top_genes: int = 3000
    n_pcs: int = 50
    obsm_key: str = "emb"


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
    """Schema for a single, discoverable parameter."""

    type: Any
    stringified_type: str
    default: Any = None
    required: bool


class TaskInfo(BaseModel):
    """Schema for all discoverable information about a single benchmark task."""

    name: str
    display_name: str
    description: str
    task_params: Dict[str, TaskParameter]
    baseline_params: Dict[str, TaskParameter]


class TaskRegistry:
    """A registry that is populated automatically as Task subclasses are defined."""

    def __init__(self):
        self._registry: Dict[str, Type["Task"]] = {}
        self._info: Dict[str, TaskInfo] = {}

    def register_task(self, task_class: type[Task]):
        """Registers a task class and introspects it to gather metadata."""
        if inspect.isabstract(task_class) or not hasattr(task_class, "display_name"):
            print(
                f"Error: Task class {task_class.__name__} missing display_name or is abstract."
            )
            return

        key = (
            getattr(task_class, "display_name", task_class.__name__)
            .lower()
            .replace(" ", "_")
        )
        self._registry[key] = task_class
        self._info[key] = self._introspect_task(task_class)

    def _stringify_type(self, annotation: Any) -> str:
        """Return a string representation of a type annotation."""
        try:
            return str(annotation).replace("typing.", "")
        except Exception:
            return str(annotation)

    def _introspect_task(self, task_class: type[Task]) -> TaskInfo:
        """Extracts parameter and metric information from a task class."""
        try:
            # 1. Get Task Parameters from the associated Pydantic input model
            task_params = {}
            if hasattr(task_class, "input_model") and issubclass(
                task_class.input_model, BaseModel
            ):
                for (
                    field_name,
                    field_info,
                ) in task_class.input_model.model_fields.items():
                    type_info = self._extract_type_info(
                        field_info.annotation, field_name
                    )
                    type_str = self._stringify_type(type_info)
                    task_params[field_name] = TaskParameter(
                        type=type_info,
                        stringified_type=type_str,
                        default=field_info.default
                        if field_info.default is not PydanticUndefined
                        else None,
                        required=field_info.is_required(),
                    )

            # 2. Get Baseline Parameters from the Pydantic baseline model
            baseline_params = {}
            if hasattr(task_class, "baseline_model") and issubclass(
                task_class.baseline_model, BaseModel
            ):
                for (
                    field_name,
                    field_info,
                ) in task_class.baseline_model.model_fields.items():
                    type_info = self._extract_type_info(
                        field_info.annotation, field_name
                    )
                    type_str = self._stringify_type(type_info)
                    baseline_params[field_name] = TaskParameter(
                        type=type_info,
                        stringified_type=type_str,
                        default=field_info.default
                        if field_info.default is not PydanticUndefined
                        else None,
                        required=field_info.is_required(),
                    )

            # 3. Get additional task metadata
            description = self._extract_description(task_class)
            display_name = getattr(task_class, "display_name", task_class.__name__)

            return TaskInfo(
                name=task_class.__name__,
                display_name=display_name,
                description=description,
                task_params=task_params,
                baseline_params=baseline_params,
            )
        except Exception as e:
            # Fallback task info if introspection fails
            print(f"Warning: Task introspection failed for {task_class.__name__}: {e}")
            return TaskInfo(
                name=task_class.__name__,
                display_name=getattr(task_class, "display_name", task_class.__name__),
                description="Task introspection failed - please check task implementation",
                task_params={},
                baseline_params={},
            )

    def _extract_type_info(self, annotation: Any, param_name: str) -> type:
        """Return the actual annotation for downstream strict type checking."""
        if annotation == inspect.Parameter.empty:
            return Any
        return annotation  # <-- Just return the annotation itself

    def _extract_description(self, task_class: Type["Task"]) -> str:
        """Extract description from task class with fallbacks."""
        # Try explicit description attribute
        if hasattr(task_class, "description"):
            return task_class.description

        # Try docstring
        doc = inspect.getdoc(task_class)
        if doc:
            # Extract first paragraph of docstring
            first_paragraph = doc.split("\n\n")[0].strip()
            return first_paragraph

        # Fallback
        return f"No description available for {task_class.__name__}"

    def list_tasks(self) -> List[str]:
        """Returns a list of all available task names."""
        return sorted(self._registry.keys())

    def get_task_info(self, task_name: str) -> TaskInfo:
        """Gets all introspected information for a given task."""
        if task_name not in self._info:
            raise ValueError(f"Task '{task_name}' not found.")
        return self._info[task_name]

    def get_task_class(self, task_name: str) -> Type["Task"]:
        """Gets the class for a given task name."""
        if task_name not in self._registry:
            available = ", ".join(self.list_tasks())
            raise ValueError(
                f"Task '{task_name}' not found. Available tasks: {available}"
            )
        return self._registry[task_name]

    # def get_task_help(self, task_name: str) -> str:
    #     """Generate detailed help text for a specific task."""
    #     try:
    #         task_info = self.get_task_info(task_name)
    #         help_text = [
    #             f"Task: {task_info.display_name}",
    #             f"Description: {task_info.description}",
    #             "",
    #         ]

    #         if task_info.task_params:
    #             help_text.append("Task Parameters:")
    #             for param_name, param_info in task_info.task_params.items():
    #                 required_str = (
    #                     "(required)"
    #                     if param_info.required
    #                     else f"(optional, default: {param_info.default})"
    #                 )
    #                 help_text.append(
    #                     f"  --{param_name.replace('_', '-')}: {param_info.type} {required_str}"
    #                 )
    #             help_text.append("")

    #         if task_info.baseline_params:
    #             if task_info.baseline_params.__name__ == "NoBaselineInput":
    #                 help_text.append("Baselines: This task does not support a baseline.")
    #             else:
    #                 help_text.append("Baseline Parameters:")
    #                 for param_name, param_info in task_info.baseline_params.items():
    #                     required_str = (
    #                         "(required)"
    #                         if param_info.required
    #                         else f"(optional, default: {param_info.default})"
    #                     )
    #                     help_text.append(
    #                         f"  --baseline-{param_name.replace('_', '-')}: {param_info.type} {required_str}"
    #                     )
    #                 help_text.append("")
    #         elif task_info.baseline_params:                
    #             help_text.append("Baseline Parameters:")
    #             for param_name, param_info in task_info.baseline_params.items():
    #                 required_str = (
    #                     "(required)"
    #                     if param_info.required
    #                     else f"(optional, default: {param_info.default})"
    #                 )
    #                 help_text.append(
    #                     f"  --baseline-{param_name.replace('_', '-')}: {param_info.type} {required_str}"
    #                 )
    #             help_text.append("")

    #         return "\n".join(help_text)

    #     except Exception as e:
    #         return f"Error generating help for task '{task_name}': {e}"

    def validate_task_input(self, task_name: str, parameters: Dict[str, Any]) -> None:
        """Strictly validate parameters using the Pydantic input model."""
        TaskClass = self.get_task_class(task_name)
        InputModel = TaskClass.input_model
        try:
            InputModel(**parameters)
        except ValidationError as e:
            raise ValueError(f"Invalid parameters for '{task_name}': {e}")
        except Exception as e:
            raise ValueError(f"Invalid parameters for '{task_name}': {e}")

    def validate_task_parameters(
        self, task_name: str, parameters: Dict[str, Any]
    ) -> List[str]:
        """Validate parameters for a task and return list of error messages."""
        errors = []
        try:
            task_info = self.get_task_info(task_name)

            # Check for unknown parameters
            known_params = set(task_info.task_params.keys())
            provided_params = set(parameters.keys())
            unknown_params = provided_params - known_params

            for param in unknown_params:
                errors.append(
                    f"Unknown parameter '{param}'. Available parameters: {list(known_params)}"
                )

            # Check for missing required parameters
            for param_name, param_info in task_info.task_params.items():
                if param_info.required and param_name not in parameters:
                    errors.append(f"Missing required parameter '{param_name}'")

        except Exception as e:
            errors.append(f"Error validating parameters: {e}")

        return errors


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
            if not all(isinstance(emb, get_args(CellRepresentation)) for emb in cell_representation):               
                raise ValueError(error_message)
            if len(cell_representation) < 2:
                raise ValueError(f"{error_message} but only one was provided")
        else:
            if not isinstance(cell_representation, get_args(CellRepresentation)):
                raise ValueError(
                    "This task requires a single cell representation for input"
                )

        return self._run_task_for_dataset(
            cell_representation,  # type: ignore
            task_input,
        )
