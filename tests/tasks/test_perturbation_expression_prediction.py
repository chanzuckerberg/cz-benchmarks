import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import anndata as ad
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.single_cell_perturbation import (
    SingleCellPerturbationDataset,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from tests.utils import create_dummy_anndata, create_dummy_perturbation_anndata


def test_perturbation_task():
    """Test that PerturbationExpressionPredictionTask executes without errors."""
    # Create dummy perturbation data
    perturbation_data: dict = create_dummy_perturbation_anndata(
        n_cells=500,
        n_genes=200,
        organism=Organism.HUMAN,
        condition_column="condition",
        split_column="split",
    )
    gene_pert = perturbation_data["gene_pert"]
    # Convert sparse matrix to dense array to avoid matrix object issues
    cell_representation = perturbation_data["adata"].X.toarray()
    # Log-normalize the data to pass validation (add small constant to avoid log(0))
    cell_representation = np.log1p(cell_representation)
    var_names = perturbation_data["adata"].var_names

    # Task and argument setup
    task = PerturbationExpressionPredictionTask()

    # Create DE results DataFrame matching expected structure
    de_results = pd.DataFrame(
        {
            "condition": [gene_pert] * len(var_names),
            "gene_id": var_names,
            "logfoldchange": np.random.randn(len(var_names)),
            "pval_adj": np.random.uniform(0, 0.01, len(var_names)),
        }
    )

    # Create masked_adata_obs DataFrame and fix condition naming
    adata = perturbation_data["adata"]
    masked_adata_obs = adata.obs.copy()

    # Fix condition naming to match task expectations
    # Task expects control cells to be named: {control_prefix}_{condition}
    control_condition_name = f"non-targeting_{gene_pert}"
    masked_adata_obs.loc[masked_adata_obs["condition"] == "ctrl", "condition"] = (
        control_condition_name
    )

    # Create target_conditions_dict dict - map condition names to lists of genes to mask
    target_conditions_dict = {}
    # Map each perturbation condition to genes to mask
    unique_conditions = np.unique(
        masked_adata_obs["condition"][
            ~masked_adata_obs["condition"].str.startswith("non-targeting")
        ]
    )
    for condition in unique_conditions:
        # Sample some genes to mask for each condition
        n_genes_to_mask = min(10, len(var_names) // 2)
        target_conditions_dict[condition] = list(
            np.random.choice(var_names, n_genes_to_mask, replace=False)
        )

    # Create AnnData with required data
    test_adata = ad.AnnData(
        X=adata.X, obs=masked_adata_obs, var=pd.DataFrame(index=var_names)
    )
    test_adata.uns["cell_barcode_condition_index"] = adata.obs.index.astype(str).values

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_results,
        gene_index=test_adata.var.index,
        cell_index=pd.Index(test_adata.uns["cell_barcode_condition_index"]),
    )

    # Five metrics per condition: accuracy, precision, recall, f1, correlation
    # We have one perturbed condition, so 5 metrics total
    num_metrics = 5

    try:
        # Test regular task execution
        results = task.run(
            cell_representation,
            task_input,
        )

        # Verify results structure - the method returns a list of MetricResult
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        assert len(results) == num_metrics

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            baseline_embedding = task.compute_baseline(
                cell_representation=cell_representation,
                baseline_type=baseline_type,
            )
            # Create a new task input with the baseline embedding
            baseline_results = task.run(baseline_embedding, task_input)
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == num_metrics
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")


def test_perturbation_expression_prediction_task_wilcoxon():
    """Test Wilcoxon path computes correct vectors and metrics."""
    # Deterministic gene set and per-condition true/predicted effects
    gene_names = ["G0", "G1", "G2", "G3"]
    true_lfc_gene_A = np.array([1.0, 0.5, -0.5, -1.0])
    true_lfc_gene_B = np.array([2.0, 1.0, -1.0, -2.0])

    # Build DE results matching the designed true effects for both conditions
    de_res_wilcoxon_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "logfoldchange": true_lfc_gene_A,
                    "target_gene": ["gene_A"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    "condition": ["gene_A"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_A"] * len(gene_names),
                }
            ),
            pd.DataFrame(
                {
                    "logfoldchange": true_lfc_gene_B,
                    "target_gene": ["gene_B"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    "condition": ["gene_B"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_B"] * len(gene_names),
                }
            ),
        ],
        ignore_index=True,
    )

    # Build AnnData with 4 groups: condition/control for A and B
    n_per_group = 4
    conditions = (
        ["gene_A"] * n_per_group
        + ["ctrl_gene_A"] * n_per_group
        + ["gene_B"] * n_per_group
        + ["ctrl_gene_B"] * n_per_group
    )
    # Create base cell names without underscores (so str.split("_").str[0] works correctly)
    base_cell_names = [f"cellbarcode{i}" for i in range(len(conditions))]
    # Create extended obs names like real dataset: base_name + "_" + condition
    obs_names = [
        f"{base_name}_{cond}" for base_name, cond in zip(base_cell_names, conditions)
    ]

    X = np.zeros((len(conditions), len(gene_names)), dtype=float)
    # Set group means so that pred_log_fc equals the designed true_lfc
    X[0:n_per_group, :] = true_lfc_gene_A  # gene_A group
    X[n_per_group : 2 * n_per_group, :] = 0.0  # ctrl_gene_A group
    X[2 * n_per_group : 3 * n_per_group, :] = true_lfc_gene_B  # gene_B group
    X[3 * n_per_group : 4 * n_per_group, :] = 0.0  # ctrl_gene_B group

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"condition": conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )
    target_conditions_dict = {"gene_A": list(gene_names), "gene_B": list(gene_names)}

    # Create log-normalized data that will pass validation and gives expected differences
    # Since the task computes: mean(treatment) - mean(control), we create data where this equals true_lfc
    base_value = 1.0  # Base value that will become 0 after log transformation
    cell_representation = np.zeros_like(X, dtype=float)

    # For treatment groups: set to base_value + true_lfc so log difference will be true_lfc
    cell_representation[0:n_per_group, :] = base_value + true_lfc_gene_A  # gene_A group
    cell_representation[n_per_group : 2 * n_per_group, :] = (
        base_value  # ctrl_gene_A group
    )
    cell_representation[2 * n_per_group : 3 * n_per_group, :] = (
        base_value + true_lfc_gene_B
    )  # gene_B group
    cell_representation[3 * n_per_group : 4 * n_per_group, :] = (
        base_value  # ctrl_gene_B group
    )

    # Add small constant to ensure fractional cell sums for validation (must exceed epsilon=1e-2)
    cell_representation += 0.003

    # Ensure de_results has the expected gene identifier column
    de_res_wilcoxon_df["gene_id"] = de_res_wilcoxon_df["names"]

    task = PerturbationExpressionPredictionTask(
        control_name="ctrl",
    )
    # Create AnnData with required data
    test_adata = adata.copy()
    test_adata.uns["cell_barcode_condition_index"] = (
        pd.Index(base_cell_names).astype(str).values
    )

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_res_wilcoxon_df,
        gene_index=test_adata.var.index,
        cell_index=pd.Index(test_adata.uns["cell_barcode_condition_index"]),
    )

    # First, check that true/pred vectors produced by _run_task match expectations
    task_output = task._run_task(cell_representation, task_input)
    assert set(task_output.pred_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert set(task_output.true_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert np.allclose(task_output.pred_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.pred_log_fc_dict["gene_B"], true_lfc_gene_B)
    assert np.allclose(task_output.true_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.true_log_fc_dict["gene_B"], true_lfc_gene_B)

    # Then, run the full task and validate metrics are perfect
    results = task.run(cell_representation, task_input)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)
    # Expect results for both conditions x 5 metric types = 10 total results
    assert len(results) == 10

    # Each result should have perfect scores
    for r in results:
        assert np.isclose(r.value, 1.0)

    # Verify metric types present
    metric_types = {result.metric_type for result in results}
    expected_types = {
        MetricType.ACCURACY_CALCULATION,
        MetricType.PRECISION_CALCULATION,
        MetricType.RECALL_CALCULATION,
        MetricType.F1_CALCULATION,
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
    }
    assert expected_types.issubset(metric_types)


def test_perturbation_expression_prediction_task_load_from_task_inputs(tmp_path):
    """Test that the task can load inputs from stored task files."""

    # Create a dummy dataset and store its task inputs
    file_path = tmp_path / "dummy_perturbation.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.obs_names = [
        "ctrl_test1_a",
        "ctrl_test2_b",
        "cond_test1_a",
        "cond_test1_b",
        "cond_test2_a",
        "cond_test2_b",
    ]
    # Provide matched control cell IDs and DE results
    adata.uns["control_cells_ids"] = {
        "test1": ["ctrl_test1_a", "ctrl_test2_b"],
        "test2": ["ctrl_test1_a", "ctrl_test2_b"],
    }
    de_conditions = ["test1"] * 10 + ["test2"] * 10
    de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
    adata.uns["de_results_wilcoxon"] = pd.DataFrame(
        {
            "condition": de_conditions,
            "gene": de_genes,
            "pval_adj": [1e-6] * 20,
            "logfoldchange": [2.0] * 20,
        }
    )
    adata.write_h5ad(file_path)

    # Create dataset and store task inputs
    dataset = SingleCellPerturbationDataset(
        path=file_path,
        organism=Organism.HUMAN,
        condition_key="condition",
        control_name="ctrl",
    )
    dataset.load_data()
    task_inputs_dir = dataset.store_task_inputs()

    # Test loading task inputs using the standalone function

    task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)

    # Verify the loaded task input has the expected structure
    assert isinstance(task_input, PerturbationExpressionPredictionTaskInput)
    assert hasattr(task_input, "adata")
    assert hasattr(task_input, "target_conditions_dict")
    assert hasattr(task_input, "de_results")
    assert isinstance(task_input.adata, ad.AnnData)
    assert isinstance(task_input.target_conditions_dict, dict)
    assert isinstance(task_input.de_results, pd.DataFrame)

    # Verify AnnData contains required data in uns
    assert "cell_barcode_condition_index" in task_input.adata.uns
    assert "control_cells_ids" in task_input.adata.uns

    # Verify data integrity
    assert task_input.de_results.shape[0] > 0
    assert task_input.adata.obs.shape[0] > 0
    assert len(task_input.adata.var.index) > 0
    assert len(task_input.target_conditions_dict) > 0

    # Verify cell barcode index matches adata size
    assert (
        len(task_input.adata.uns["cell_barcode_condition_index"])
        == dataset.adata.shape[0]
    )


def test_perturbation_expression_prediction_task_with_shuffled_input():
    """Test that the perturbation task works with shuffled AnnData input."""

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=4,  # Use 4 genes for easier testing
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.obs_names = [
            "ctrl_test1_a",
            "ctrl_test2_b",
            "cond_test1_a",
            "cond_test1_b",
            "cond_test2_a",
            "cond_test2_b",
        ]
        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {
            "test1": ["ctrl_test1_a", "ctrl_test2_b"],
            "test2": ["ctrl_test1_a", "ctrl_test2_b"],
        }

        # Create DE results that match the actual gene names in the dataset
        # The create_dummy_anndata creates genes with names like "ENSG000000000{i:02d}"
        gene_names = (
            adata.var.index.tolist()
        )  # Get actual gene names from the created dataset

        # Create sufficient DE results for both conditions using actual gene names
        de_data = []
        for condition in ["test1", "test2"]:
            for gene in gene_names:
                de_data.append(
                    {
                        "condition": condition,
                        "gene_id": gene,  # Use gene_id instead of gene
                        "pval_adj": 1e-6,
                        "logfoldchange": 2.0,
                    }
                )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and load data with relaxed parameters
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=1.0,  # Use all genes to avoid filtering
            min_de_genes_to_mask=1,  # Minimum threshold
            pval_threshold=1.0,  # Accept all p-values
            min_logfoldchange=0.0,  # Accept all log fold changes
        )
        dataset.load_data()

        # Create task input with original ordering
        original_task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
            gene_index=dataset.control_matched_adata.var.index,
            cell_index=pd.Index(
                dataset.control_matched_adata.uns["cell_barcode_condition_index"]
            ),
        )

        # Create shuffled version of the AnnData
        shuffled_adata = dataset.control_matched_adata.copy()

        # Shuffle obs and var with fixed random seed for reproducibility
        np.random.seed(42)
        obs_shuffled = shuffled_adata.obs.sample(frac=1, random_state=42)
        var_shuffled = shuffled_adata.var.sample(frac=1, random_state=42)

        # Get the indices for shuffling
        obs_order = [shuffled_adata.obs.index.get_loc(i) for i in obs_shuffled.index]
        var_order = [shuffled_adata.var.index.get_loc(i) for i in var_shuffled.index]

        # Shuffle the data matrix accordingly
        X_shuffled = shuffled_adata.X[np.ix_(obs_order, var_order)]

        # Create new AnnData with shuffled data
        shuffled_adata = ad.AnnData(X=X_shuffled, obs=obs_shuffled, var=var_shuffled)

        # Copy over the uns data
        shuffled_adata.uns = dataset.control_matched_adata.uns.copy()

        # Create task input with shuffled AnnData
        shuffled_task_input = PerturbationExpressionPredictionTaskInput(
            adata=shuffled_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
            gene_index=shuffled_adata.var.index,
            cell_index=pd.Index(shuffled_adata.uns["cell_barcode_condition_index"]),
        )

        # Create identical model output for both (using original dimensions)
        model_output = np.random.rand(
            dataset.control_matched_adata.shape[0],
            dataset.control_matched_adata.shape[1],
        )

        # For shuffled input, we need to reorder the model output to match
        model_output_shuffled = model_output[np.ix_(obs_order, var_order)]

        # Initialize task
        task = PerturbationExpressionPredictionTask()

        # Run task with both inputs
        original_result = task._run_task(model_output, original_task_input)
        shuffled_result = task._run_task(model_output_shuffled, shuffled_task_input)

        # Results should be identical (same conditions, same relative data)
        assert set(original_result.pred_log_fc_dict.keys()) == set(
            shuffled_result.pred_log_fc_dict.keys()
        )
        assert set(original_result.true_log_fc_dict.keys()) == set(
            shuffled_result.true_log_fc_dict.keys()
        )

        # The predicted values should be the same since we used correspondingly shuffled model output
        for condition in original_result.pred_log_fc_dict.keys():
            np.testing.assert_allclose(
                original_result.pred_log_fc_dict[condition],
                shuffled_result.pred_log_fc_dict[condition],
                rtol=1e-10,
                err_msg=f"Predicted log fold changes differ for condition {condition}",
            )
            np.testing.assert_allclose(
                original_result.true_log_fc_dict[condition],
                shuffled_result.true_log_fc_dict[condition],
                rtol=1e-10,
                err_msg=f"True log fold changes differ for condition {condition}",
            )


def test_perturbation_task_apply_model_ordering():
    """Test the apply_model_ordering method for PerturbationExpressionPredictionTaskInput."""

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=4,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.obs_names = [
            "ctrl_test1_a",
            "ctrl_test2_b",
            "cond_test1_a",
            "cond_test1_b",
            "cond_test2_a",
            "cond_test2_b",
        ]

        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {
            "test1": ["ctrl_test1_a", "ctrl_test2_b"],
            "test2": ["ctrl_test1_a", "ctrl_test2_b"],
        }

        # Create DE results
        gene_names = adata.var.index.tolist()
        de_data = []
        for condition in ["test1", "test2"]:
            for gene in gene_names:
                de_data.append(
                    {
                        "condition": condition,
                        "gene_id": gene,
                        "pval_adj": 1e-6,
                        "logfoldchange": 2.0,
                    }
                )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and get task input
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            min_de_genes_to_mask=1,  # Lower threshold so genes get sampled
            percent_genes_to_mask=1.0,  # Use all genes
        )
        dataset.load_data()

        # Create task input
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
            gene_index=dataset.control_matched_adata.var.index,
            cell_index=pd.Index(
                dataset.control_matched_adata.uns["cell_barcode_condition_index"]
            ),
        )

        # Create model data with shuffled ordering (same content, different order)
        # We need to create model data that has the same cells as in cell_barcode_condition_index
        np.random.seed(42)  # For reproducible test

        # Get the cell barcodes that should match between model and task input
        task_cell_barcodes = task_input.adata.uns["cell_barcode_condition_index"]
        task_genes = task_input.adata.var.index

        # Create model AnnData with the same cells and genes but in shuffled order
        gene_order = np.random.permutation(task_genes)
        cell_order = np.random.permutation(task_cell_barcodes)

        # Create a model AnnData with the shuffled ordering
        model_adata = ad.AnnData(
            X=np.random.rand(len(cell_order), len(gene_order)),
            obs=pd.DataFrame(index=cell_order),
            var=pd.DataFrame(index=gene_order),
        )

        # Store original orderings
        original_gene_order = task_input.adata.var.index.copy()
        original_cell_barcode_index = task_input.adata.uns[
            "cell_barcode_condition_index"
        ].copy()

        # Apply model ordering
        task_input.gene_index = model_adata.var.index
        task_input.cell_index = model_adata.obs.index

        # Verify that orderings have changed to match model data
        pd.testing.assert_index_equal(task_input.gene_index, model_adata.var.index)
        np.testing.assert_array_equal(
            task_input.cell_index,
            model_adata.obs.index.astype(str).values,
        )

        # Verify orderings are different from original (unless by chance they're the same)
        assert not task_input.gene_index.equals(original_gene_order)
        assert not np.array_equal(task_input.cell_index, original_cell_barcode_index)


def test_perturbation_task_apply_model_ordering_validation():
    """Test that apply_model_ordering validates matching gene and cell sets."""

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=4,
            n_genes=3,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1"]
        adata.obs_names = ["ctrl_a", "ctrl_b", "cond_a", "cond_b"]

        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {"test1": ["ctrl_a", "ctrl_b"]}

        # Create DE results
        gene_names = adata.var.index.tolist()
        de_data = []
        for gene in gene_names:
            de_data.append(
                {
                    "condition": "test1",
                    "gene_id": gene,
                    "pval_adj": 1e-6,
                    "logfoldchange": 2.0,
                }
            )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and get task input
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            min_de_genes_to_mask=1,  # Lower threshold so genes get sampled
            percent_genes_to_mask=1.0,  # Use all genes
        )
        dataset.load_data()

        # Create task input
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
            gene_index=dataset.control_matched_adata.var.index,
            cell_index=pd.Index(
                dataset.control_matched_adata.uns["cell_barcode_condition_index"]
            ),
        )

        # Test with mismatched genes
        model_adata_bad_genes = create_dummy_anndata(
            n_cells=4,
            n_genes=3,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        model_adata_bad_genes.obs_names = task_input.adata.obs.index  # Same cells
        model_adata_bad_genes.var_names = [
            "different_gene1",
            "different_gene2",
            "different_gene3",
        ]  # Different genes
        task = PerturbationExpressionPredictionTask(
            control_name="ctrl",
        )

        with pytest.raises(
            ValueError, match="Model data contains genes that are not in the task input"
        ):
            task_input.gene_index = model_adata_bad_genes.var.index
            task_input.cell_index = model_adata_bad_genes.obs.index
            task._run_task(
                np.random.rand(len(task_input.cell_index), len(task_input.gene_index)),
                task_input,
            )

        # Test with mismatched cells
        model_adata_bad_cells = task_input.adata.copy()
        model_adata_bad_cells.obs_names = [
            "different_cell1",
            "different_cell2",
            "different_cell3",
            "different_cell4",
        ]  # Different cells
        task_input.cell_index = model_adata_bad_cells.obs.index
        with pytest.raises(
            ValueError, match="Model data contains genes that are not in the task input"
        ):
            task._run_task(
                np.random.rand(len(task_input.cell_index), len(task_input.gene_index)),
                task_input,
            )
