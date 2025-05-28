import collections
from unittest import mock

import pytest

from czbenchmarks.cli.utils import aggregate_task_results
from czbenchmarks.cli.types import TaskResult, DatasetDetail, ModelDetail
from czbenchmarks.metrics.types import AggregatedMetricResult, MetricResult
from czbenchmarks.models.types import ModelType


@mock.patch(
    "czbenchmarks.cli.utils.metric_utils.aggregate_results",
    return_value=[mock.MagicMock(spec=AggregatedMetricResult)],
)
def test_aggregate_task_results_same_key(mock_agg_metrics):
    dummy_metrics = [
        mock.MagicMock(spec=MetricResult) for _ in range(4)
    ]  # there are separate tests for aggregating metrics so mock them
    task_results = [
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}
            ),
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[:2],
            runtime_metrics={},
        ),
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}
            ),
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[2:],
            runtime_metrics={},
        ),
    ]
    agg_results = aggregate_task_results(task_results)

    assert len(agg_results) == 1
    assert agg_results[0].task_name == "dummy"
    assert agg_results[0].task_name_display == "Dummy"
    assert agg_results[0].model.type == ModelType.SCVI
    assert agg_results[0].model.args == {"model_variant": "homo_sapiens"}
    assert agg_results[0].datasets == [
        DatasetDetail(name="dummy", organism="homo_sapiens")
    ]
    assert agg_results[0].metrics == mock_agg_metrics.return_value
    mock_agg_metrics.assert_called_once_with(dummy_metrics)


@mock.patch(
    "czbenchmarks.cli.utils.metric_utils.aggregate_results",
    return_value=[mock.MagicMock(spec=AggregatedMetricResult)],
)
def test_aggregate_task_results_different_key(mock_agg_metrics):
    dummy_metrics = [
        mock.MagicMock(spec=MetricResult) for _ in range(12)
    ]  # there are separate tests for aggregating metrics so mock them
    task_results = [
        # all of these should aggregrate separately
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}
            ),
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[:2],
            runtime_metrics={},
        ),
        TaskResult(
            task_name="extra",  # different task
            task_name_display="Extra",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}
            ),
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[2:4],
            runtime_metrics={},
        ),
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCGPT, args={"model_variant": "human"}
            ),  # different model
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[4:6],
            runtime_metrics={},
        ),
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "mus_musculus"}
            ),  # different model args
            datasets=[DatasetDetail(name="dummy", organism="homo_sapiens")],
            metrics=dummy_metrics[6:8],
            runtime_metrics={},
        ),
        TaskResult(
            task_name="dummy",
            task_name_display="Dummy",
            model=ModelDetail(
                type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}
            ),
            datasets=[
                DatasetDetail(name="extended", organism="homo_sapiens")
            ],  # different dataset name
            metrics=dummy_metrics[8:10],
            runtime_metrics={},
        ),
    ]
    agg_results = aggregate_task_results(task_results)

    assert len(agg_results) == 5
    assert {"dummy": 4, "extra": 1} == collections.Counter(
        [r.task_name for r in agg_results]
    )
    assert {ModelType.SCVI: 4, ModelType.SCGPT: 1} == collections.Counter(
        [r.model.type for r in agg_results]
    )
    assert {"dummy": 4, "extended": 1} == collections.Counter(
        [ds.name for r in agg_results for ds in r.datasets]
    )
    assert all(r.metrics == mock_agg_metrics.return_value for r in agg_results)
    mock_agg_metrics.assert_has_calls(
        [
            mock.call(dummy_metrics[0:2]),
            mock.call(dummy_metrics[2:4]),
            mock.call(dummy_metrics[4:6]),
            mock.call(dummy_metrics[6:8]),
            mock.call(dummy_metrics[8:10]),
        ],
        any_order=True,
    )


def test_aggregate_task_results_empty():
    assert aggregate_task_results([]) == []


def test_aggregate_task_results_errors():
    tr = TaskResult(
        task_name="dummy",
        task_name_display="Dummy",
        model=ModelDetail(type=ModelType.SCVI, args={"model_variant": "homo_sapiens"}),
        datasets=[],
        metrics=[],
        runtime_metrics={"gpu_mem": "123GiB"},
    )
    with pytest.raises(ValueError):
        aggregate_task_results([tr])
