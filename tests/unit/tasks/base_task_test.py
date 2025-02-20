from typing import Dict
from unittest.mock import MagicMock

import pytest
from czibench.datasets.base import BaseDataset
from czibench.datasets.types import DataType
from czibench.tasks.base import BaseTask


class TestTask(BaseTask):
    @property
    def required_inputs(self):
        return {DataType.ANNDATA}

    @property
    def required_outputs(self):
        return {DataType.METADATA}

    def _run_task(self, data: BaseDataset) -> BaseDataset:
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {"accuracy": 0.95}


@pytest.fixture
def mock_task():
    return TestTask()


@pytest.fixture
def base_dataset():
    dataset = MagicMock(spec=BaseDataset)
    dataset.inputs = {DataType.ANNDATA: MagicMock()}
    dataset.outputs = {DataType.METADATA: MagicMock()}
    return dataset


def test_validate_success(mock_task, base_dataset):
    mock_task.validate(base_dataset)


def test_validate_missing_inputs(mock_task, base_dataset):
    base_dataset.inputs = {}
    with pytest.raises(ValueError, match="Missing required inputs"):
        mock_task.validate(base_dataset)


def test_validate_missing_outputs(mock_task, base_dataset):
    base_dataset.outputs = {}
    with pytest.raises(ValueError, match="Missing required outputs"):
        mock_task.validate(base_dataset)


def test_run(mock_task, base_dataset):
    data, results = mock_task.run(base_dataset)
    assert data == base_dataset
    assert results == {"accuracy": 0.95}
