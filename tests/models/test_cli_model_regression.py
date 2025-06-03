import pytest
import json
from pathlib import Path
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.cli.cli_run import (
    run_with_inference,
    ModelArgs,
    TaskArgs,
    write_results,
    CacheOptions,
)
from czbenchmarks.tasks import ClusteringTask, PerturbationTask
from datetime import datetime
import numpy as np
from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.types import DataType
from czbenchmarks.runner import run_inference


@pytest.mark.skipif(
    not pytest.run_model_tests,
    reason="Model regression tests skipped. Use --run-model-tests to run them.",
)
def test_model_regression(
    model_name, variant, dataset_name, task_name, tolerance_percent
):
    """
    Model regression test for CLI end-to-end workflow.

    This tests:
    1. A single benchmark (i.e. a single dataset & task) against a baseline file
    2. Generates both baseline and results files in tests/fixtures/baselines/
       - baseline file: {model}_{variant}_{dataset}_{task}_baseline.json
       - results file: {model}_{variant}_{dataset}_{task}_results.json

    The test ensures that model outputs remain consistent with previously established baselines.
    If the baseline file doesn't exist, it will be created for future comparisons.

    Command line arguments:
    --run-model-tests: Required flag to run these tests (skipped by default)
    --tolerance-percent: Optional float (default: 0.2) for metric comparison tolerance
    """
    # region Setup
    dataset = load_dataset(dataset_name)
    assert dataset is not None, f"Failed to load dataset {dataset_name}"

    # Create model args with the variant
    args = {}
    if variant:
        args["model_variant"] = [variant]

    if task_name == "perturbation":
        gene_pert = "CREB1+ctrl"
        args["gene_pert"] = [gene_pert]
        task_args = [
            TaskArgs(
                name=task_name,
                task=PerturbationTask(),
                set_baseline=True,
                baseline_args={"gene_pert": gene_pert},
            ),
        ]
    elif task_name == "clustering":
        task_args = [
            TaskArgs(
                name=task_name,
                task=ClusteringTask(label_key="cell_type"),
                set_baseline=True,
                baseline_args={},
            ),
        ]
    else:
        raise ValueError(f"Invalid task name: {task_name}")

    model_args = [ModelArgs(name=model_name, args=args)]

    cache_options = CacheOptions(
        download_embeddings=False,
        upload_embeddings=False,
        upload_results=False,
        remote_cache_url="",
    )
    # endregion

    # region Run Inference and Task
    task_results = run_with_inference(
        dataset_names=[dataset_name],
        model_args=model_args,
        task_args=task_args,
        cache_options=cache_options,
    )
    # endregion

    # region Save and Compare Results
    baseline_dir = Path("tests/fixtures/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    variant_suffix = f"_{variant}" if variant else ""
    baseline_file = (
        baseline_dir
        / f"{model_name}{variant_suffix}_{dataset_name}_{task_name}_baseline.json"
    )
    results_file = (
        baseline_dir
        / f"{model_name}{variant_suffix}_{dataset_name}_{task_name}_results.json"
    )

    # Run inference and get processed dataset
    processed_dataset = run_inference(model_name, dataset)

    if not baseline_file.exists():
        write_results(
            task_results,
            output_format="json",
            output_file=str(baseline_file),
            cache_options=cache_options,
        )
        pytest.fail(
            f"Baseline file {baseline_file} did not exist and was created. Please review and commit this file."
        )
    write_results(
        task_results,
        output_format="json",
        output_file=str(results_file),
        cache_options=cache_options,
    )

    # Save embeddings
    embeddings_file = (
        baseline_dir / f"{model_name}{variant_suffix}_{dataset_name}_embeddings.npy"
    )
    if not embeddings_file.exists():
        try:
            model_type = ModelType[model_name]
            embeddings = processed_dataset.get_output(model_type, DataType.EMBEDDING)
            np.save(embeddings_file, embeddings)
        except Exception as e:
            print(f"Could not extract/save embeddings for {model_name}: {e}")

    if not results_file.exists():
        pytest.fail(f"Results file {results_file} did not exist.")
    if baseline_file.exists():
        with (
            open(results_file) as actual_results,
            open(baseline_file) as expected_results,
        ):
            actual_json = json.load(actual_results)
            expected_json = json.load(expected_results)

            # Ignore fields that are not relevant for comparison
            for ignored_field in ["czbenchmarks_version", "args", "timestamp"]:
                actual_json.pop(ignored_field, None)
                expected_json.pop(ignored_field, None)

            # Compare metrics with percentage tolerance
            compare_metrics(
                actual_json,
                expected_json,
                tolerance_percent=tolerance_percent,
                results_file=results_file,
                baseline_file=baseline_file,
            )
    else:
        raise ValueError(f"Baseline file {baseline_file} does not exist")
    # endregion

    # region Sanity Checks
    assert task_results is not None, "Task results should not be None"
    assert len(task_results) > 0, "Should have at least one task result"
    assert actual_json, "Actual results JSON should not be empty"
    assert expected_json, "Expected results JSON should not be empty"
    # endregion

    # Clean up temporary results file
    if results_file.exists():
        results_file.unlink()


def compare_metrics(
    actual,
    expected,
    path="root",
    tolerance_percent=0.1,
    results_file=None,
    baseline_file=None,
):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            assert key in actual, f"Missing key '{key}' in actual results at {path}"
            compare_metrics(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                tolerance_percent=tolerance_percent,
                results_file=results_file,
                baseline_file=baseline_file,
            )
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(actual) == len(expected), f"List length mismatch at {path}"
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare_metrics(
                a,
                e,
                path=f"{path}[{index}]",
                tolerance_percent=tolerance_percent,
                results_file=results_file,
                baseline_file=baseline_file,
            )
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
                f"Metric at {path} differs by {percent_diff * 100:.4f}% (actual: {actual}, expected: {expected}) "
                f"which is greater than allowed tolerance {tolerance_percent * 100:.2f}%.\nCheck {results_file} vs {baseline_file}"
            )
    elif isinstance(expected, datetime) and isinstance(actual, datetime):
        assert actual == expected, (
            f"Value mismatch at {path}: actual={actual}, expected={expected} in {results_file} vs {baseline_file}"
        )
    else:
        assert actual == expected, (
            f"Value mismatch at {path}: actual={actual}, expected={expected} in {results_file} vs {baseline_file}"
        )
