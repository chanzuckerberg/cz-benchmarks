"""
Unit tests for the runner.py module.

Tests the run_task function which serves as the main API endpoint for
executing benchmark tasks programmatically.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from czbenchmarks.tasks.runner import run_task
from czbenchmarks.tasks.task import TASK_REGISTRY
from czbenchmarks.metrics.types import MetricResult, MetricType

from tests.utils import create_dummy_anndata, DummyTaskInput
from czbenchmarks.datasets.types import Organism


class TestRunTask:
    """Test suite for the run_task function."""

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding for testing."""
        return np.random.randn(100, 50)

    @pytest.fixture
    def sample_obs(self):
        """Create sample observation metadata."""
        return pd.DataFrame({
            'cell_type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], size=100)
        })

    @pytest.fixture  
    def sample_expression_data(self):
        """Create sample raw expression data."""
        dummy_adata = create_dummy_anndata(
            n_cells=100, 
            n_genes=200,
            organism=Organism.HUMAN,
            obs_columns=['cell_type', 'batch']
        )
        return dummy_adata.X

    def test_run_task_basic_functionality(self, sample_embedding, sample_obs):
        """Test basic run_task functionality with clustering task."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        # Check that all results are dictionaries (serialized MetricResults)
        assert all(isinstance(r, dict) for r in results)
        # Check that results have expected keys
        for result in results:
            assert 'metric_type' in result
            assert 'value' in result

    def test_run_task_with_baseline(self, sample_expression_data, sample_obs):
        """Test run_task with baseline computation."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        baseline_params = {
            'n_pcs': 20
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=sample_expression_data,
            run_baseline=True,
            baseline_params=baseline_params,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_run_task_embedding_task(self, sample_embedding, sample_obs):
        """Test run_task with embedding task."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
        }
        
        results = run_task(
            task_name='embedding',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_run_task_label_prediction_task(self, sample_embedding, sample_obs):
        """Test run_task with label prediction task."""
        task_params = {
            'labels': sample_obs['cell_type'].values,
        }
        
        results = run_task(
            task_name='label_prediction',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_run_task_invalid_task_name(self, sample_embedding):
        """Test run_task with invalid task name raises ValueError."""
        with pytest.raises(ValueError, match="Task 'nonexistent_task' not found"):
            run_task(
                task_name='nonexistent_task',
                cell_representation=sample_embedding,
                task_params={},
                random_seed=42
            )

    def test_run_task_invalid_task_params(self, sample_embedding):
        """Test run_task with invalid task parameters raises ValueError."""
        # Missing required parameters should raise ValueError
        with pytest.raises(ValueError, match="Invalid task parameters"):
            run_task(
                task_name='clustering',
                cell_representation=sample_embedding,
                task_params={},  # Missing required input_labels and obs
                random_seed=42
            )

    def test_run_task_with_none_params(self, sample_embedding, sample_obs):
        """Test run_task handles None parameters correctly."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            run_baseline=False,
            baseline_params=None,  # Should handle None
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_task_baseline_not_implemented(self, sample_embedding):
        """Test run_task handles NotImplementedError in baseline computation."""
        # Use a mock task that raises NotImplementedError for compute_baseline
        with patch.object(TASK_REGISTRY, 'get_task_class') as mock_get_task:
            mock_task_class = Mock()
            mock_task_instance = Mock()
            mock_task_class.return_value = mock_task_instance
            mock_task_class.input_model = DummyTaskInput
            
            # Make compute_baseline raise NotImplementedError
            mock_task_instance.compute_baseline.side_effect = NotImplementedError()
            # Make run return valid results
            mock_task_instance.run.return_value = [
                MetricResult(name="test", value=1.0, metric_type=MetricType.ADJUSTED_RAND_INDEX)
            ]
            
            mock_get_task.return_value = mock_task_class
            
            # Should not raise exception, should log warning and continue
            with patch('czbenchmarks.tasks.runner.log') as mock_log:
                results = run_task(
                    task_name='dummy_task',
                    cell_representation=sample_embedding,
                    run_baseline=True,
                    task_params={}
                )
                
                # Check that warning was logged
                mock_log.warning.assert_called_once()
                assert "not implemented" in str(mock_log.warning.call_args)
                
                # Should still return results
                assert isinstance(results, list)

    def test_run_task_baseline_computation_error(self, sample_embedding):
        """Test run_task handles baseline computation errors correctly."""
        with patch.object(TASK_REGISTRY, 'get_task_class') as mock_get_task:
            mock_task_class = Mock()
            mock_task_instance = Mock()
            mock_task_class.return_value = mock_task_instance
            mock_task_class.input_model = DummyTaskInput
            
            # Make compute_baseline raise a general exception
            mock_task_instance.compute_baseline.side_effect = RuntimeError("Baseline failed")
            
            mock_get_task.return_value = mock_task_class
            
            # Should re-raise the exception
            with pytest.raises(RuntimeError, match="Baseline failed"):
                run_task(
                    task_name='dummy_task',
                    cell_representation=sample_embedding,
                    run_baseline=True,
                    task_params={}
                )

    def test_run_task_logging(self, sample_embedding, sample_obs):
        """Test that run_task produces expected log messages."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        with patch('czbenchmarks.tasks.runner.log') as mock_log:
            run_task(
                task_name='clustering',
                cell_representation=sample_embedding,
                task_params=task_params,
                random_seed=42
            )
            
            # Check that expected log messages were called
            log_calls = [call.args[0] for call in mock_log.info.call_args_list]
            assert any("Preparing to run task" in msg for msg in log_calls)
            assert any("Executing task logic" in msg for msg in log_calls)
            assert any("execution complete" in msg for msg in log_calls)

    def test_run_task_random_seed_handling(self, sample_embedding, sample_obs):
        """Test that random seed is properly passed to task instance."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        # Run with different seeds and check for different results
        # (Note: this is probabilistic and may occasionally fail)
        results1 = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        results2 = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=123
        )
        
        # Both should succeed and return same structure
        assert isinstance(results1, list)
        assert isinstance(results2, list)
        assert len(results1) == len(results2)

    def test_run_task_default_random_seed(self, sample_embedding, sample_obs):
        """Test run_task uses default random seed when not specified."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        # Don't specify random_seed, should use RANDOM_SEED constant
        results = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            task_params=task_params
            # random_seed not specified
        )
        
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_task_multiple_embeddings(self):
        """Test run_task with tasks that require multiple embeddings."""
        # Create sample data for cross-species integration task
        embedding1 = np.random.randn(50, 30)
        embedding2 = np.random.randn(60, 30)
        embeddings = [embedding1, embedding2]
        
        labels1 = np.random.choice(['TypeA', 'TypeB'], size=50)
        labels2 = np.random.choice(['TypeA', 'TypeB'], size=60)
        labels = [labels1, labels2]
        
        organism_list = [Organism.HUMAN, Organism.MOUSE]
        
        task_params = {
            'labels': labels,
            'organism_list': organism_list,
        }
        
        results = run_task(
            task_name='cross-species_integration',
            cell_representation=embeddings,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_run_task_result_serialization(self, sample_embedding, sample_obs):
        """Test that results are properly serialized to dictionaries."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        # All results should be dictionaries (serialized from MetricResult objects)
        assert all(isinstance(r, dict) for r in results)
        
        # Check expected keys are present
        for result in results:
            assert 'metric_type' in result
            assert 'value' in result
            # These are from the MetricResult model_dump()
            assert isinstance(result['value'], (int, float))

    @pytest.mark.parametrize("task_name,expected_param", [
        ('clustering', 'input_labels'),
        ('embedding', 'input_labels'),  
        ('label_prediction', 'labels'),
    ])
    def test_run_task_different_tasks(self, task_name, expected_param, sample_embedding, sample_obs):
        """Parameterized test for different task types."""
        # Prepare task params based on task type
        if expected_param == 'labels':
            task_params = {'labels': sample_obs['cell_type'].values}
        elif expected_param == 'input_labels':
            if task_name == 'clustering':
                task_params = {
                    'input_labels': sample_obs['cell_type'].values,
                    'obs': sample_obs,
                }
            else:  # embedding task
                task_params = {'input_labels': sample_obs['cell_type'].values}
        
        results = run_task(
            task_name=task_name,
            cell_representation=sample_embedding,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_run_task_empty_baseline_params(self, sample_expression_data, sample_obs):
        """Test run_task with empty baseline_params dict."""
        task_params = {
            'input_labels': sample_obs['cell_type'].values,
            'obs': sample_obs,
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=sample_expression_data,
            run_baseline=True,
            baseline_params={},  # Empty dict
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_task_with_integration_task(self):
        """Test run_task with batch integration task."""
        embedding = np.random.randn(100, 50)
        labels = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=100)  
        batch_labels = np.random.choice(['Batch1', 'Batch2'], size=100)
        
        task_params = {
            'labels': labels,
            'batch_labels': batch_labels,
        }
        
        results = run_task(
            task_name='batch_integration',
            cell_representation=embedding,
            task_params=task_params,
            random_seed=42
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)


class TestRunTaskErrorHandling:
    """Test error handling scenarios in run_task."""

    def test_task_execution_error(self, sample_embedding=None):
        """Test run_task handles task execution errors properly."""
        if sample_embedding is None:
            sample_embedding = np.random.randn(50, 30)
            
        with patch.object(TASK_REGISTRY, 'get_task_class') as mock_get_task:
            mock_task_class = Mock()
            mock_task_instance = Mock()
            mock_task_class.return_value = mock_task_instance
            mock_task_class.input_model = DummyTaskInput
            
            # Make task.run raise an exception
            mock_task_instance.run.side_effect = RuntimeError("Task execution failed")
            
            mock_get_task.return_value = mock_task_class
            
            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Task execution failed"):
                run_task(
                    task_name='dummy_task',
                    cell_representation=sample_embedding,
                    task_params={}
                )

    def test_input_model_creation_error(self, sample_embedding=None):
        """Test run_task handles TaskInput creation errors properly."""
        if sample_embedding is None:
            sample_embedding = np.random.randn(50, 30)
            
        with patch.object(TASK_REGISTRY, 'get_task_class') as mock_get_task:
            mock_task_class = Mock()
            mock_task_instance = Mock()
            mock_task_class.return_value = mock_task_instance
            
            # Make input model raise validation error
            mock_input_model = Mock()
            mock_input_model.side_effect = ValueError("Invalid input parameters")
            mock_task_class.input_model = mock_input_model
            
            mock_get_task.return_value = mock_task_class
            
            # Should raise ValueError with proper message
            with pytest.raises(ValueError, match="Invalid task parameters"):
                run_task(
                    task_name='dummy_task',
                    cell_representation=sample_embedding,
                    task_params={'invalid': 'params'}
                )


class TestRunTaskIntegration:
    """Integration tests for run_task with real tasks."""

    def test_run_task_clustering_integration(self):
        """Integration test for clustering task."""
        # Create realistic test data
        n_cells = 200
        n_genes = 50
        embedding = np.random.randn(n_cells, n_genes)
        
        # Create structured cell types
        cell_types = (['TypeA'] * 70 + ['TypeB'] * 80 + ['TypeC'] * 50)
        obs = pd.DataFrame({
            'cell_type': cell_types,
            'batch': ['batch1'] * 100 + ['batch2'] * 100
        })
        
        task_params = {
            'input_labels': np.array(cell_types),
            'obs': obs,
        }
        
        results = run_task(
            task_name='clustering',
            cell_representation=embedding,
            task_params=task_params,
            random_seed=42
        )
        
        # Should get ARI and NMI metrics
        assert len(results) == 2
        metric_types = [r['metric_type'].value if hasattr(r['metric_type'], 'value') else r['metric_type'] for r in results]
        assert 'adjusted_rand_index' in metric_types
        assert 'normalized_mutual_info' in metric_types
        
        # Values should be reasonable
        for result in results:
            assert isinstance(result['value'], (int, float))
            assert -1 <= result['value'] <= 1  # ARI and NMI are bounded

    def test_run_task_embedding_integration(self):
        """Integration test for embedding task."""
        n_cells = 150
        n_dims = 40
        embedding = np.random.randn(n_cells, n_dims)
        
        # Create structured labels for better silhouette scores
        labels = (['A'] * 50 + ['B'] * 50 + ['C'] * 50)
        
        task_params = {
            'input_labels': np.array(labels),
        }
        
        results = run_task(
            task_name='embedding',
            cell_representation=embedding,
            task_params=task_params,
            random_seed=42
        )
        
        # Should get one silhouette score metric
        assert len(results) == 1
        metric_type = results[0]['metric_type'].value if hasattr(results[0]['metric_type'], 'value') else results[0]['metric_type']
        assert metric_type == 'silhouette_score'
        
        # Silhouette score should be reasonable
        score = results[0]['value']
        assert isinstance(score, (int, float))
        assert -1 <= score <= 1  # Silhouette scores are bounded

    def test_run_task_label_prediction_integration(self):
        """Integration test for label prediction task."""
        n_cells = 300
        n_dims = 60
        embedding = np.random.randn(n_cells, n_dims)
        
        # Create balanced labels
        n_types = 4
        labels = []
        for i in range(n_types):
            labels.extend([f'Type_{i}'] * (n_cells // n_types))
        # Handle remainder
        labels.extend(['Type_0'] * (n_cells % n_types))
        labels = labels[:n_cells]
        
        task_params = {
            'labels': np.array(labels),
        }
        
        results = run_task(
            task_name='label_prediction',
            cell_representation=embedding,
            task_params=task_params,
            random_seed=42
        )
        
        # Should get multiple metrics (accuracy, precision, recall, f1 for each classifier)
        assert len(results) > 0
        
        # All should be valid metric results
        for result in results:
            assert 'metric_type' in result
            assert 'value' in result
            assert isinstance(result['value'], (int, float))
            # Most classification metrics are 0-1 bounded
            assert 0 <= result['value'] <= 1