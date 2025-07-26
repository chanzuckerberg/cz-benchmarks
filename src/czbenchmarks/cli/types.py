from typing import Any, Optional, Type
from pydantic import BaseModel
from ..tasks.task import Task


class TaskParameter(BaseModel):
    """A runtime-discovered representation of a command-line parameter."""

    name: str
    type: Any
    help: str
    required: bool = False
    default: Optional[Any] = None
    is_flag: bool = False


class TaskDefinition(BaseModel):
    """A simplified, declarative definition of a benchmark task for the registry."""

    model_config = {"arbitrary_types_allowed": True}

    task_class: Type[Task]
    input_model: Type[BaseModel]
    display_name: str
    description: str
    requires_multiple_datasets: bool = False
