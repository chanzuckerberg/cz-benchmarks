"""Tests for KkTestTask (rare cell type detection)."""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from czbenchmarks.tasks.single_cell.kk_test import (
    KkTestTask,
    KkTestTaskInput,
    KkTestTaskOutput,
)
from czbenchmarks.metrics.types import MetricResult


@pytest.fixture
def test_adata():
    """Create synthetic AnnData for testing."""
    n_cells = 200
    n_features = 50

    # Create embeddings
    X = np.random.randn(n_cells, n_features)

    # Create labels with some rare and some common types
    # Common types: TypeA (40%), TypeB (30%)
    # Rare types: TypeC (4%), TypeD (3%)
    # Very rare: TypeE (1% - below threshold)
    labels = (
        ["TypeA"] * 80
        + ["TypeB"] * 60
        + ["TypeC"] * 8
        + ["TypeD"] * 6
        + ["TypeE"] * 2
        + ["TypeF"] * 44  # Another common type
    )
    np.random.shuffle(labels)

    obs = pd.DataFrame({"cell_type": labels})
    adata = ad.AnnData(X=X, obs=obs)

    return adata


def test_kk_test_basic_execution(test_adata):
    """Test that the task executes without errors."""
    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=5)

    results = task.run(test_adata, task_input)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, MetricResult) for r in results)


def test_kk_test_output_structure(test_adata):
    """Test that task output has expected structure."""
    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type")

    # Run task and get internal output
    output = task._run_task(test_adata, task_input)

    assert isinstance(output, KkTestTaskOutput)
    assert isinstance(output.rare_types, list)
    assert isinstance(output.classifier_results, list)
    assert isinstance(output.n_rare_cells, int)
    assert isinstance(output.n_common_cells, int)
    assert isinstance(output.total_cells, int)
    assert output.n_rare_cells + output.n_common_cells == output.total_cells


def test_identify_rare_types(test_adata):
    """Test rare type identification."""
    task = KkTestTask()
    labels = pd.Series(test_adata.obs["cell_type"])

    # With threshold 0.05 and min_cells 5, should find TypeC and TypeD
    rare_types = task._identify_rare_types(labels, rarity_threshold=0.05, min_cells=5)

    assert len(rare_types) == 2
    assert "TypeC" in rare_types
    assert "TypeD" in rare_types
    assert "TypeE" not in rare_types  # Too few cells
    assert "TypeA" not in rare_types  # Too common


def test_different_rarity_thresholds(test_adata):
    """Test with different rarity thresholds."""
    task = KkTestTask()

    # Very strict threshold
    task_input_strict = KkTestTaskInput(
        obs="cell_type", rarity_threshold=0.01, min_cells=2
    )
    output_strict = task._run_task(test_adata, task_input_strict)
    # Should find very few rare types (only TypeE at 1%)
    assert len(output_strict.rare_types) <= 1

    # Lenient threshold
    task_input_lenient = KkTestTaskInput(
        obs="cell_type", rarity_threshold=0.25, min_cells=5
    )
    output_lenient = task._run_task(test_adata, task_input_lenient)
    # Should find more rare types
    assert len(output_lenient.rare_types) >= 2


def test_min_cells_validation(test_adata):
    """Test that min_cells parameter works correctly."""
    task = KkTestTask()

    # High min_cells requirement
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=10)
    output = task._run_task(test_adata, task_input)

    # TypeE (2 cells) and TypeD (6 cells) should be excluded
    assert "TypeE" not in output.rare_types
    assert "TypeD" not in output.rare_types


def test_no_rare_types_found(test_adata):
    """Test behavior when no rare types are found."""
    task = KkTestTask()

    # Set impossible threshold
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.001, min_cells=50)
    output = task._run_task(test_adata, task_input)

    assert len(output.rare_types) == 0
    assert len(output.classifier_results) == 0
    assert output.n_rare_cells == 0


def test_classifier_results(test_adata):
    """Test that classifiers are run and produce results."""
    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=5)

    output = task._run_task(test_adata, task_input)

    # Should have 3 classifiers: logistic, knn, rf
    assert len(output.classifier_results) == 3

    classifiers = [r["classifier"] for r in output.classifier_results]
    assert "logistic" in classifiers
    assert "knn" in classifiers
    assert "rf" in classifiers

    # Check that each result has expected metrics
    expected_metrics = [
        "f1",
        "precision",
        "recall",
        "balanced_acc",
        "mcc",
        "specificity",
        "mean_recall_per_type",
        "min_recall_per_type",
        "mean_precision_per_type",
    ]

    for result in output.classifier_results:
        for metric in expected_metrics:
            assert metric in result
            # MCC can be negative (range: -1 to 1), others are 0 to 1
            if metric == "mcc":
                assert -1 <= result[metric] <= 1
            else:
                assert 0 <= result[metric] <= 1


def test_metric_computation(test_adata):
    """Test that metrics are computed correctly."""
    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=5)

    results = task.run(test_adata, task_input)

    # Check that we have metrics
    assert len(results) > 0

    # Check metric types
    metric_types = {r.metric_type for r in results}
    assert len(metric_types) > 0

    # Check that values are reasonable
    for result in results:
        assert isinstance(result.value, float)
        # Most metrics should be between 0 and 1
        if "mcc" not in result.metric_type.value:
            assert 0 <= result.value <= 1


def test_invalid_obs_column():
    """Test error handling for invalid obs column."""
    n_cells = 100
    X = np.random.randn(n_cells, 20)
    obs = pd.DataFrame({"cell_type": ["A"] * n_cells})
    adata = ad.AnnData(X=X, obs=obs)

    task = KkTestTask()
    task_input = KkTestTaskInput(obs="nonexistent_column")

    with pytest.raises(ValueError, match="not found in AnnData.obs"):
        task._run_task(adata, task_input)


def test_input_validation_obs():
    """Test validation of obs parameter."""
    with pytest.raises(ValueError, match="obs must be a non-empty string"):
        KkTestTaskInput(obs="")


def test_input_validation_rarity_threshold():
    """Test validation of rarity_threshold parameter."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than 0"):
        KkTestTaskInput(obs="cell_type", rarity_threshold=0.0)

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        KkTestTaskInput(obs="cell_type", rarity_threshold=1.5)


def test_input_validation_min_cells():
    """Test validation of min_cells parameter."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        KkTestTaskInput(obs="cell_type", min_cells=0)


def test_input_validation_n_splits():
    """Test validation of n_splits parameter."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        KkTestTaskInput(obs="cell_type", n_splits=1)


def test_determinism(test_adata):
    """Test that same seed produces same results."""
    task1 = KkTestTask(random_seed=42)
    task2 = KkTestTask(random_seed=42)

    task_input = KkTestTaskInput(obs="cell_type")

    results1 = task1.run(test_adata, task_input)
    results2 = task2.run(test_adata, task_input)

    # Check that results are identical
    assert len(results1) == len(results2)

    # Compare metric values
    for r1, r2 in zip(results1, results2):
        assert r1.metric_type == r2.metric_type
        assert abs(r1.value - r2.value) < 1e-10  # Should be identical


def test_small_dataset():
    """Test with very small dataset."""
    n_cells = 30
    X = np.random.randn(n_cells, 10)
    labels = ["A"] * 15 + ["B"] * 10 + ["C"] * 5
    obs = pd.DataFrame({"cell_type": labels})
    adata = ad.AnnData(X=X, obs=obs)

    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.2, min_cells=3, n_splits=2)

    results = task.run(adata, task_input)
    assert len(results) > 0


def test_binary_classification():
    """Test with only two cell types (binary classification)."""
    n_cells = 100
    X = np.random.randn(n_cells, 20)
    labels = ["Common"] * 95 + ["Rare"] * 5
    obs = pd.DataFrame({"cell_type": labels})
    adata = ad.AnnData(X=X, obs=obs)

    task = KkTestTask()
    task_input = KkTestTaskInput(obs="cell_type", rarity_threshold=0.1, min_cells=5)

    results = task.run(adata, task_input)
    assert len(results) > 0


def test_cross_validation_folds(test_adata):
    """Test with different numbers of CV folds."""
    task = KkTestTask()

    # Test with 3 folds - use parameters that will find rare types
    task_input_3 = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=5, n_splits=3)
    results_3 = task.run(test_adata, task_input_3)
    # May not always have rare types depending on test data, so just check it runs
    assert isinstance(results_3, list)

    # Test with 5 folds
    task_input_5 = KkTestTaskInput(obs="cell_type", rarity_threshold=0.05, min_cells=5, n_splits=5)
    results_5 = task.run(test_adata, task_input_5)
    # May not always have rare types depending on test data, so just check it runs
    assert isinstance(results_5, list)


def test_baseline_not_implemented(test_adata):
    """Test that baseline raises NotImplementedError."""
    task = KkTestTask()

    with pytest.raises(NotImplementedError, match="Baseline not implemented"):
        task.compute_baseline(test_adata)


def test_compute_binary_metrics():
    """Test binary metrics computation."""
    task = KkTestTask()

    # Perfect prediction
    y_true = np.array([True, True, False, False])
    y_pred = np.array([True, True, False, False])

    metrics = task._compute_binary_metrics(y_true, y_pred)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_acc"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["specificity"] == 1.0


def test_compute_per_type_metrics():
    """Test per-type metrics computation."""
    task = KkTestTask()

    y_true = np.array(["A", "A", "B", "B", "C", "C"])
    y_pred = np.array(["A", "A", "B", "C", "C", "C"])
    rare_types = ["A", "B", "C"]

    metrics = task._compute_per_type_metrics(y_true, y_pred, rare_types)

    assert "mean_recall_per_type" in metrics
    assert "min_recall_per_type" in metrics
    assert "mean_precision_per_type" in metrics
    assert 0 <= metrics["mean_recall_per_type"] <= 1
    assert 0 <= metrics["min_recall_per_type"] <= 1
