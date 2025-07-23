from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from ..tasks.task import Task

class TaskParameter(BaseModel):
    """Defines a command-line parameter for a task."""
    name: str
    type: Any
    help: str
    required: bool = False
    default: Optional[Any] = None
    is_flag: bool = False

class TaskDefinition(BaseModel):
    """A complete, declarative definition of a benchmark task."""
    model_config = {"arbitrary_types_allowed": True}

    task_class: Type[Task]
    input_model: Type[BaseModel]
    display_name: str
    description: str
    parameters: List[TaskParameter]
    baseline_parameters: List[TaskParameter] = Field(default_factory=list)
    requires_multiple_datasets: bool = False