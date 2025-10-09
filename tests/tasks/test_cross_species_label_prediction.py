import pytest
import numpy as np
from czbenchmarks.tasks.single_cell.cross_species_label_prediction import (
    CrossSpeciesLabelPredictionTask,
    CrossSpeciesLabelPredictionTaskInput,
)
from czbenchmarks.tasks.utils import aggregate_cells_to_samples
from czbenchmarks.datasets.types import Organism
from czbenchmarks.metrics.types import MetricResult


@pytest.fixture
def cross_species_test_data():
    """Create dummy data for cross-species testing."""
    n_cells_human = 100
    n_cells_mouse = 80
    n_features = 20  # embedding dimension

    human_embeddings = np.random.randn(n_cells_human, n_features)
    human_labels = np.random.choice(["healthy", "disease"], size=n_cells_human)
    human_sample_ids = [f"human_sample_{i // 10}" for i in range(n_cells_human)]

    mouse_embeddings = np.random.randn(n_cells_mouse, n_features)
    mouse_labels = np.random.choice(["healthy", "disease"], size=n_cells_mouse)
    mouse_sample_ids = [f"mouse_sample_{i // 8}" for i in range(n_cells_mouse)]

    return {
        "human_embeddings": human_embeddings,
        "human_labels": human_labels,
        "human_sample_ids": human_sample_ids,
        "mouse_embeddings": mouse_embeddings,
        "mouse_labels": mouse_labels,
        "mouse_sample_ids": mouse_sample_ids,
    }


def test_sample_aggregation(cross_species_test_data):
    """Test the sample aggregation functionality."""
    embeddings = cross_species_test_data["human_embeddings"]
    labels = cross_species_test_data["human_labels"]
    sample_ids = cross_species_test_data["human_sample_ids"]

    sample_emb, sample_labels, sample_ids_out = aggregate_cells_to_samples(
        embeddings, labels, sample_ids, aggregation_method="mean"
    )

    # Check that we have fewer samples than cells
    assert len(sample_emb) < len(embeddings)
    assert len(sample_emb) == len(sample_labels)
    assert len(sample_emb) == len(sample_ids_out)

    # Check that sample embeddings have correct shape
    assert sample_emb.shape[1] == embeddings.shape[1]

    # do the same thing with median to make sure we get different values
    sample_emb_median, _, _ = aggregate_cells_to_samples(
        embeddings, labels, sample_ids, aggregation_method="median"
    )

    # Should have same number of samples but different values
    assert sample_emb_median.shape == sample_emb.shape
    assert not np.array_equal(sample_emb_median, sample_emb)

    # Test error handling for mismatched lengths
    with pytest.raises(ValueError, match="Mismatched lengths"):
        aggregate_cells_to_samples(
            embeddings[:-1],
            labels,
            sample_ids,  # One fewer embedding row
        )


def test_cross_species_classification(cross_species_test_data):
    """Test cross-species classification functionality."""
    task = CrossSpeciesLabelPredictionTask()

    human_emb = cross_species_test_data["human_embeddings"]
    human_labels = cross_species_test_data["human_labels"]
    mouse_emb = cross_species_test_data["mouse_embeddings"]
    mouse_labels = cross_species_test_data["mouse_labels"]

    results = task._run_cross_species_classification(
        human_emb, human_labels, mouse_emb, mouse_labels, "human", "mouse"
    )

    # Should have 3 classifiers (lr, knn, rf)
    assert len(results) == 3

    numeric_keys = {"accuracy", "f1", "precision", "recall", "auroc"}
    other_keys = {"classifier", "train_species", "test_species"}
    for result in results:
        assert (numeric_keys | other_keys) == result.keys()
        assert all(0 <= result[key] <= 1 for key in numeric_keys)


def test_cross_species_task_cell_level(cross_species_test_data):
    """Test full cross-species task execution at cell level (no aggregation)."""
    task = CrossSpeciesLabelPredictionTask()

    human_embeddings = cross_species_test_data["human_embeddings"]
    mouse_embeddings = cross_species_test_data["mouse_embeddings"]
    human_labels = cross_species_test_data["human_labels"]
    mouse_labels = cross_species_test_data["mouse_labels"]

    task_input = CrossSpeciesLabelPredictionTaskInput(
        labels=[human_labels, mouse_labels],
        organisms=[Organism.HUMAN, Organism.MOUSE],
        aggregation_method="none",
    )

    results = task.run(
        cell_representation=[human_embeddings, mouse_embeddings],
        task_input=task_input,
    )

    assert len(results) > 0
    assert all(isinstance(r, MetricResult) for r in results)


def test_cross_species_task_sample_level(cross_species_test_data):
    """Test cross-species task execution with sample-level aggregation."""
    task = CrossSpeciesLabelPredictionTask()

    human_embeddings = cross_species_test_data["human_embeddings"]
    mouse_embeddings = cross_species_test_data["mouse_embeddings"]
    human_labels = cross_species_test_data["human_labels"]
    mouse_labels = cross_species_test_data["mouse_labels"]
    human_sample_ids = cross_species_test_data["human_sample_ids"]
    mouse_sample_ids = cross_species_test_data["mouse_sample_ids"]

    task_input = CrossSpeciesLabelPredictionTaskInput(
        labels=[human_labels, mouse_labels],
        organisms=[Organism.HUMAN, Organism.MOUSE],
        sample_ids=[human_sample_ids, mouse_sample_ids],
        aggregation_method="mean",
        n_folds=2,
    )

    results = task.run(
        cell_representation=[human_embeddings, mouse_embeddings],
        task_input=task_input,
    )

    assert all(isinstance(r, MetricResult) for r in results)


def test_invalid_inputs(cross_species_test_data):
    """Test error handling for invalid inputs."""
    task = CrossSpeciesLabelPredictionTask()

    human_embeddings = cross_species_test_data["human_embeddings"]
    mouse_embeddings = cross_species_test_data["mouse_embeddings"]
    human_labels = cross_species_test_data["human_labels"]
    mouse_labels = cross_species_test_data["mouse_labels"]

    # Test wrong number of species
    task_input = CrossSpeciesLabelPredictionTaskInput(
        labels=[human_labels],  # Only one species
        organisms=[Organism.HUMAN],
        aggregation_method="none",
    )

    with pytest.raises(ValueError):
        task.run([human_embeddings], task_input)

    # Test mismatched lengths
    task_input = CrossSpeciesLabelPredictionTaskInput(
        labels=[human_labels, mouse_labels],
        organisms=[Organism.HUMAN, Organism.MOUSE],
        aggregation_method="none",
    )

    with pytest.raises(ValueError):
        task.run([human_embeddings], task_input)  # Only one embedding

    # Test missing sample_ids when aggregation != "none"
    task_input = CrossSpeciesLabelPredictionTaskInput(
        labels=[human_labels, mouse_labels],
        organisms=[Organism.HUMAN, Organism.MOUSE],
        aggregation_method="mean",  # Requires sample_ids
        sample_ids=None,
    )

    with pytest.raises(ValueError):
        task.run([human_embeddings, mouse_embeddings], task_input)


def test_baseline_not_implemented():
    """Test that baseline raises NotImplementedError."""
    import numpy as np
    task = CrossSpeciesLabelPredictionTask()
    
    # Create dummy expression data
    expression_data = np.random.rand(10, 5)

    with pytest.raises(NotImplementedError, match="Baseline not implemented"):
        task.compute_baseline(expression_data)
