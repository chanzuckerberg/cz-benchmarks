import argparse
from functools import cached_property
import operator
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, computed_field

from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.metrics.types import AggregatedMetricResult, MetricResult
from czbenchmarks.models.types import ModelType
from czbenchmarks.models import utils as model_utils
from czbenchmarks.tasks.base import BaseTask


TaskType = TypeVar("TaskType", bound=BaseTask)
ModelArgsDict = dict[str, str | int]  # Arguments passed to model inference
RuntimeMetricsDict = dict[
    str, str | int | float
]  # runtime metrics like elapsed time or CPU count, not implemented yet


class ModelArgs(BaseModel):
    name: str  # Upper-case model name e.g. SCVI
    args: dict[str, list[str | int]]  # Args forwarded to the model container


class TaskArgs(BaseModel, Generic[TaskType]):
    model_config = {"arbitrary_types_allowed": True}  # Required to support TaskType
    name: str  # Lower-case task name e.g. embedding
    task: TaskType
    compute_baseline: bool
    baseline_args: dict[str, Any]


class DatasetDetail(BaseModel):
    name: str
    organism: str

    @cached_property
    def _display_info(self) -> tuple[str, str]:
        return dataset_utils.dataset_to_display(self.name)

    @computed_field
    @property
    def name_display(self) -> str:
        return self._display_info[0]

    @computed_field
    @property
    def subset_display(self) -> str:
        return self._display_info[1]


class ModelDetail(BaseModel):
    type: ModelType
    args: ModelArgsDict

    @cached_property
    def _display_info(self) -> tuple[str, str]:
        return model_utils.model_to_display(self.type, self.args)

    @computed_field
    @property
    def name_display(self) -> str:
        return self._display_info[0]

    @computed_field
    @property
    def variant_display(self) -> str:
        return self._display_info[1]


class TaskResult(BaseModel):
    task_name: str
    task_name_display: str
    model: ModelDetail
    datasets: list[DatasetDetail]
    metrics: list[MetricResult | AggregatedMetricResult]
    runtime_metrics: RuntimeMetricsDict = {}  # not implementing any of these for now

    @property
    def aggregation_key(self) -> str:
        """return a key based on the task, model, and datasets in order to aggregate the same results together"""
        datasets = ",".join(
            (ds.name for ds in sorted(self.datasets, key=operator.attrgetter("name")))
        )
        model_args = "_".join(
            (f"{key}-{value!s}" for key, value in sorted(self.model.args.items()))
        )
        return f"{self.task_name}|{self.model.type}({model_args})|{datasets}"


class CacheOptions(BaseModel):
    download_embeddings: bool
    upload_embeddings: bool
    upload_results: bool
    remote_cache_url: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CacheOptions":
        remote_cache_url = args.remote_cache_url or ""
        return cls(
            remote_cache_url=remote_cache_url,
            download_embeddings=bool(remote_cache_url)
            and args.remote_cache_download_embeddings,
            upload_embeddings=bool(remote_cache_url)
            and args.remote_cache_upload_embeddings,
            upload_results=bool(remote_cache_url) and args.remote_cache_upload_results,
        )
