import pytest
import json
from pathlib import Path
from czbenchmarks.models import utils as model_utils
from czbenchmarks.tasks import utils as task_utils
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs, write_results
from czbenchmarks.tasks import ClusteringTask
from unittest.mock import patch, MagicMock

# TODO: Make this dynamic by reading from model configs
MODEL_VARIANT_TEST_CASES = {
    "SCGPT": ["human"],
    "SCVI": ["homo_sapiens"],
    "GENEFORMER": ["gf_6L_30M"],
    "SCGENEPT": ["scgpt__adamson"],
    "UCE": ["4l"],
    "TRANSCRIPTFORMER": ["tf-sapiens"]
}

@pytest.mark.parametrize(
    "model_name,variant,dataset_name,task_name",
    [
        (model, variant, dataset, task)
        for model in model_utils.list_available_models()
        for variant in MODEL_VARIANT_TEST_CASES[model]
        for dataset in ["human_spermatogenesis"] # human organism is currently supported by all models
        for task in ["clustering"] # task_utils.TASK_NAMES
    ]
)
def test_model_regression(model_name, variant, dataset_name, task_name, mock_container_runner):
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
    write_results(task_results, output_format="json", output_file=str(results_file))
    
    if baseline_file.exists():
        with open(results_file) as actual_results, open(baseline_file) as expected_results:
            actual_json = json.load(actual_results)
            expected_json = json.load(expected_results)
            assert actual_json == expected_json, f"Results differ from baseline. Check {results_file} vs {baseline_file}"
    else:
        raise ValueError(f"Baseline file {baseline_file} does not exist")
    #endregion
    
    #region Sanity Checks
    assert task_results is not None, "Task results should not be None"
    assert len(task_results) > 0, "Should have at least one task result"
    assert actual_json is not None, "Actual results JSON should not be None"
    assert expected_json is not None, "Expected results JSON should not be None"
    assert len(actual_json) > 0, "Actual results JSON should not be empty"
    assert len(expected_json) > 0, "Expected results JSON should not be empty"
    #endregion

    # Clean up temporary results file
    if results_file.exists():
        results_file.unlink() 
