from typing import List, Set
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.datasets import DataType
from czbenchmarks.metrics.types import MetricResult
from czbenchmarks.models.types import ModelType

class SimpleTask(BaseTask):
    def __init__(self, my_param: int = 10):
        self.my_param = my_param

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data, model_type: ModelType):
        # Retrieve necessary data
        self.embedding = data.get_output(model_type, DataType.EMBEDDING)
        self.labels = data.get_input(DataType.METADATA)["cell_type"]
        # Implement your task logic here
        # e.g., calculate some predictions or transform embeddings

    def _compute_metrics(self) -> List[MetricResult]:
        # Compute and return a list of MetricResult objects
        metric_value = 1.0
        return [MetricResult(metric_type="simple_metric", value=metric_value)]