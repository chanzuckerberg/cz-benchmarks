from pydantic import BaseModel


class TaskInput(BaseModel):
    """Base class for task inputs."""
    model_config = {"arbitrary_types_allowed": True}


class MetricInput(BaseModel):
    """Base class for metric inputs."""
    model_config = {"arbitrary_types_allowed": True}


class TaskOutput(BaseModel):
    """Base class for task outputs."""
    model_config = {"arbitrary_types_allowed": True} 