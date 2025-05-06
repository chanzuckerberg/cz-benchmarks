from czbenchmarks.models import utils as model_utils
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.tasks import utils as task_utils
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs, write_results
from czbenchmarks.tasks import (
    ClusteringTask,
)
import pytest
import json
from pathlib import Path
from czbenchmarks.datasets import Organism


# TODO: Make this dynamic by reading from model configs
ORGANISM = Organism.HUMAN
MODEL_VARIANTS = {
    "SCGPT": ["human"],
    "SCVI": ["homo_sapiens", "mus_musculus"],
    "GENEFORMER": ["gf_12L_30M", "gf_12L_95M", "gf_20L_95M", "gf_6L_30M"],
    "SCGENEPT": [
        "scgenept_go_all_gpt_concat__adamson",
        "scgenept_go_c_gpt_concat__adamson",
        "scgenept_go_f_gpt_concat__adamson",
        "scgenept_go_p_gpt_concat__adamson",
        "scgenept_ncbi+uniprot_gpt__adamson",
        "scgenept_ncbi_gpt__adamson",
        "scgpt__adamson",
        "scgenept_go_all_gpt_concat__norman",
        "scgenept_go_c_gpt_concat__norman",
        "scgenept_go_f_gpt_concat__norman",
        "scgenept_go_p_gpt_concat__norman",
        "scgenept_ncbi+uniprot_gpt__norman",
        "scgenept_ncbi_gpt__norman",
        "scgpt__norman"
    ], 
    "UCE": ["33l", "4l"],
    "TRANSCRIPTFORMER": ["tf-sapiens", "tf-exemplar", "tf-metazoa"]  # From docker/transcriptformer/model.py
}

@pytest.mark.parametrize(
    "model_name,dataset_name,task_name",
    [
        (model, dataset, task)
        for model in model_utils.list_available_models()
        for variant in MODEL_VARIANTS[model]
        for dataset in ["human_spermatogenesis"] # dataset_utils.list_available_datasets()
        for task in ["clustering"] # task_utils.TASK_NAMES
    ]
)
def test_model_regression(model_name, variant, dataset_name, task_name):
    """
    Model regression test for CLI end-to-end workflow.
    
    This test:
    1. A single benchmark (i.e. a single dataset & task) against a baseline file
    2. Generates both baseline and results files in tests/fixtures/baselines/
       - baseline file: {model}_{dataset}_{task}_baseline.json
       - results file: {model}_{dataset}_{task}_results.json
    
    The test ensures that model outputs remain consistent with previously established baselines.
    If the baseline file doesn't exist, it will be created for future comparisons.
    """
    #region Setup
    dataset = load_dataset(dataset_name)
    assert dataset is not None, f"Failed to load dataset {dataset_name}"

    model_args = [ModelArgs(name=model_name, args={})]
    
    task_args = [
        TaskArgs(
            name=task_name,
            task=ClusteringTask(label_key="cell_type"),
            set_baseline=True,  # Set to True to generate baseline results
        ),
    ]
    #endregion

    #region Run Inference and Task
    task_results = run_with_inference(
        dataset_names=[dataset_name],
        model_args=model_args,
        task_args=task_args,
    )
    #endregion
    
    #region Save and Compare Results
    baseline_dir = Path("tests/fixtures/baselines")
    baseline_file = baseline_dir / f"{model_name}_{dataset_name}_{task_name}_baseline.json"
    results_file = baseline_dir / f"{model_name}_{dataset_name}_{task_name}_results.json"
    
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
    assert any(result.model_type == model_name for result in task_results), f"No results found for model {model_name}"
    #endregion

    # Clean up temporary results file
    if results_file.exists():
        results_file.unlink() 
