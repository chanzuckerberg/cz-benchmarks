import pytest
import json
from pathlib import Path
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs, write_results
from czbenchmarks.tasks import ClusteringTask
from unittest.mock import patch, MagicMock
from datetime import datetime

MODEL_VARIANT_TEST_CASES = [
    ("SCGPT", "human"),
    ("SCVI", "homo_sapiens"),
    ("GENEFORMER", "gf_6L_30M"),
    ("SCGENEPT", "scgpt__adamson"),
    ("UCE", "4l"),
    ("TRANSCRIPTFORMER", "tf-sapiens"),
]

@pytest.mark.parametrize(
    "model_name,variant,dataset_name,task_name",
    [
        (model, variant, dataset, task)
        for model, variant in MODEL_VARIANT_TEST_CASES
        for dataset in ["human_spermatogenesis"]
        for task in ["clustering"]
    ]
)
def test_model_regression(model_name, variant, dataset_name, task_name, tolerance_percent, mock_container_runner):
    """
    Model regression test for CLI end-to-end workflow.
    
    This tests:
    1. A single benchmark (i.e. a single dataset & task) against a baseline file
    2. Generates both baseline and results files in tests/fixtures/baselines/
       - baseline file: {model}_{variant}_{dataset}_{task}_baseline.json
       - results file: {model}_{variant}_{dataset}_{task}_results.json
    
    The test ensures that model outputs remain consistent with previously established baselines.
    If the baseline file doesn't exist, it will be created for future comparisons.
    """
    #region Setup
    dataset = load_dataset(dataset_name)
    assert dataset is not None, f"Failed to load dataset {dataset_name}"

    # Create model args with the variant
    model_args = [ModelArgs(
        name=model_name,
        args={
            "model_variant": [variant] if variant else []
        }
    )]
    
    task_args = [
        TaskArgs(
            name=task_name,
            task=ClusteringTask(label_key="cell_type"),
            set_baseline=True,  # Set to True to generate baseline results
            baseline_args={},  # Add empty baseline args
        ),
    ]
    #endregion

    #region Run Inference and Task
    if mock_container_runner:  # --mock-container-runner argument passed to test on the command line
        # Mock the ContainerRunner class to avoid nvidia runtime error if not available
        mock_runner = MagicMock()
        mock_runner.run.return_value = dataset
        with patch('czbenchmarks.runner.ContainerRunner', return_value=mock_runner):
            task_results = run_with_inference(
                dataset_names=[dataset_name],
                model_args=model_args,
                task_args=task_args,
            )
    else:
        task_results = run_with_inference(
            dataset_names=[dataset_name],
            model_args=model_args,
            task_args=task_args,
        )
    #endregion
    
    #region Save and Compare Results
    baseline_dir = Path("tests/fixtures/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    variant_suffix = f"_{variant}" if variant else ""
    baseline_file = baseline_dir / f"{model_name}{variant_suffix}_{dataset_name}_{task_name}_baseline.json"
    results_file = baseline_dir / f"{model_name}{variant_suffix}_{dataset_name}_{task_name}_results.json"
    
    if not baseline_file.exists():
        write_results(task_results, output_format="json", output_file=str(baseline_file))
        pytest.fail(f"Baseline file {baseline_file} did not exist and was created. Please review and commit this file.")
    write_results(task_results, output_format="json", output_file=str(results_file))
    
    if baseline_file.exists():
        with open(results_file) as actual_results, open(baseline_file) as expected_results:
            actual_json = json.load(actual_results)
            expected_json = json.load(expected_results)
            # Compare metrics with percentage tolerance
            compare_metrics(actual_json, expected_json, tolerance_percent=tolerance_percent, results_file=results_file, baseline_file=baseline_file)
    else:
        raise ValueError(f"Baseline file {baseline_file} does not exist")
    #endregion
    
    #region Sanity Checks
    assert task_results is not None, "Task results should not be None"
    assert len(task_results) > 0, "Should have at least one task result"
    assert actual_json, "Actual results JSON should not be empty"
    assert expected_json, "Expected results JSON should not be empty"
    #endregion

    # Clean up temporary results file
    if results_file.exists():
        results_file.unlink()

def compare_metrics(actual, expected, path="root", tolerance_percent=0.1, results_file=None, baseline_file=None):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            assert key in actual, f"Missing key '{key}' in actual results at {path}"
            compare_metrics(actual[key], expected[key], path=f"{path}.{key}", tolerance_percent=tolerance_percent, results_file=results_file, baseline_file=baseline_file)
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(actual) == len(expected), f"List length mismatch at {path}"
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare_metrics(a, e, path=f"{path}[{index}]", tolerance_percent=tolerance_percent, results_file=results_file, baseline_file=baseline_file)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            diff = abs(actual - expected)
            assert diff <= tolerance_percent, (
                f"Metric at {path} differs by {diff} (actual: {actual}, expected: {expected}) "
                f"which is greater than allowed absolute tolerance {tolerance_percent} for zero baseline.\nCheck {results_file} vs {baseline_file}"
            )
        else:
            percent_diff = abs(actual - expected) / abs(expected)
            assert percent_diff <= tolerance_percent, (
                f"Metric at {path} differs by {percent_diff*100:.4f}% (actual: {actual}, expected: {expected}) "
                f"which is greater than allowed tolerance {tolerance_percent*100:.2f}%.\nCheck {results_file} vs {baseline_file}"
            )
    elif isinstance(expected, datetime) and isinstance(actual, datetime):
        assert actual == expected, (
            f"Value mismatch at {path}: actual={actual}, expected={expected} in {results_file} vs {baseline_file}"
        )
    else:
        assert actual == expected, (
            f"Value mismatch at {path}: actual={actual}, expected={expected} in {results_file} vs {baseline_file}"
        )
