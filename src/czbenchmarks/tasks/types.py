from pydantic import BaseModel


class TaskInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class MetricInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid" 