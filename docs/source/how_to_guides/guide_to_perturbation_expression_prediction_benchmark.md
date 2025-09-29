# Guide to Perturbation Expression Prediction Benchmark

## Overview

This task evaluates a model's ability to predict expression for masked genes using the remaining (unmasked) genes for a given cell as context. Genes are randomly selected for masking from the set of genes that are identified as differentially expressed, based on threshold parameters explained in [Masking Parameters](#masking-parameters). For Replogle K562 Essentials[^replogle-k562-essentials], the provided controls used for the differential expression (DE) analysis have been determined based on GEM group and UMI count and are stored in the dataset along with the DE results.

- Single cell perturbation datasets contain perturbed and control cells. Matched controls have been determined for each condition and are stored in the unstructured portion of the AnnData under the key `control_cell_ids`. 
- The differential expression results are also stored in the unstructured portion of the AnnData in the keys `de_results_wilcoxon` for analysis that utilized the Wilcoxon rank-sum test. Currently, this is the only DE selection method that is supported, but additional options are planned. 
- The gene expression values in the dataset are counts, as provided by the authors. Additional preprocessing, such as scaling and log transformation, should be performed by the user.

This benchmark is designed for evaluation by any model that outputs a cell × gene prediction matrix that can be aligned along the cell and gene axes with the input adata can be used with the task. The task ensures alignment by validating gene and cell indices against the dataset. The predictions provided to the task are expected to be **log fold change**. # FIXME MICHELLE check task input expectations

## Dataset Functionality and Parameters

The data loading method accomplishes the following:

- Perturbed cells and their matched controls are selected and indexed to create a new AnnData object for each condition. Each condition is then added to the AnnData `obs` metadata column defined by the parameter ``{condition_key}``.
- In the matched data, the perturbations are labeled as ``{perturb}``, and matched control cells are labeled as ``{control_name}_{perturb}``, where ``{perturb}`` is the name of the perturbation and ``{control_name}`` is a configurable parameter.
- Combinatorial perturbations are not currently supported.
- For each condition, a subset of DE genes are sampled from the DE results and masked from their default values. These become the prediction targets for the model.
- The objective is to predict the masked expression values for the prediction targets per cell/condition.

### Data Parameters

The following parameters are used in loading the data:

- `condition_key`: The name of the column in `adata.obs` and the DE results containing condition labels for perturbations and controls. Default is "condition".
- `control_name`: The name used to denote control samples and to form control labels ``{control_name}_{perturb}``. Default is "ctrl".
- `de_gene_col`: The name of the column in the DE results indicating gene identifiers to be considered for masking. Default is "gene_id".

### Masking Parameters

The following parameters control masking of the DE genes:

- `percent_genes_to_mask`: The fraction of DE genes per condition to mask as prediction targets for the model. Default value is 0.5.
- `min_de_genes_to_mask`: Minimum number of sampled DE genes required for a condition to be eligible for masking. This threshold is applied after the genes are sampled. Default value is 5.
- `pval_threshold`: Maximum adjusted p-value for DE filtering based on the output of the DE analysis. This data must be in the column `pval_adj`. Default value is 1e-4.
- `min_logfoldchange`: Minimum absolute log-fold change to determine when a gene is considered differentially expressed. This data must be in the column `logfoldchange`. Only used when the DE analysis uses Wilcoxon rank-sum. Default value is 1.
- `condition_key`: Key for the column in `adata.obs` specifying conditions (default value is "condition")
- `control_name`: Name of the control condition (default value is "ctrl")

Parameters shared with other single-cell datasets (e.g., `path`, `organism`, `task_inputs_dir`) are also required but not listed here.

### Saving the Dataset

To reload and reuse datasets without re-running preprocessing, the outputs of the dataset can be saved with:

```python
task_inputs_dir = dataset.store_task_inputs()
```

## Task Functionality and Parameters 

This task evaluates perturbation-induced expression predictions against their ground truth values by calculating predicted and experimental log fold change (LFC) values. The predictions provided to the task are expected to be **log fold change**. Predicted log fold changes are computed per condition as the difference in mean expression between perturbed and matched control cells, for the subset of masked genes. # FIXME MICHELLE check task input expectations

The class also calculates a baseline prediction (`compute_baseline` method), which takes as input a `baseline_type`, either `median` (default) or `mean`, that calculates the median or mean expression values, respectively, across all cells in the dataset.

The following parameters are used by the task, via the `PerturbationExpressionPredictionTaskInput` class:  

- `adata`: The complete AnnData object containing control-matched and masked data.
- `de_results`: the DE results filtered by the dataset class(`SingleCellPerturbationDataset`) according to user provided thresholds.
- `target_conditions_dict`: Dictionary of target conditions whose genes were randomly selected for masking.
- `cell_index`: Sequence of user-provided cell barcodes vertically aligned with `cell_representation` matrix, which contains the predictions from the model.
- `gene_index`: Sequence of user-provided gene names horizontally aligned with `cell_representation` matrix, which contains the predictions from the model.

If the dataset results were saved as above, they can be loaded with:

```python
task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)
```

### Notes on Loading Model Predictions

When a user loads in model predictions, the cells and genes whose expression values are predicted should each be a subset of those in the dataset. At the start of the task, validation is performed to ensure these criteria are met. 

It is essential that the mapping of the cells (rows) and genes (columns) from the model expression predictions to those in the dataset is correct. Thus, the `PerturbationExpressionPredictionTaskInput` requires a `gene_index` and `cell_index` to be provided by the user.

## Metrics

The task produces a per-condition correlation by comparing predicted and ground-truth log fold change (LFC) values for the masked genes. The comparison metric is:

- **Spearman correlation (rank)**: Rank correlation between the raw predicted and ground truth LFC values.


Results are generated for each perturbation condition separately. Downstream reporting may aggregate scores across conditions (e.g., mean and standard deviation).

For large-scale benchmarks, metrics can be exported to CSV/JSON via the provided `czbenchmarks.tasks.utils.print_metrics_summary helper`, or integrated into custom logging frameworks.

## Example Usage

For example use cases, see the example script `examples/example_perturbation_expression_prediction.py`. 

[^replogle-k562-essentials]: Replogle, J. M., Elgamal, R. M., Abbas, A. et al. Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. Cell, 185(14):2559–2575.e28 (2022). [DOI](https://doi.org/10.1016/j.cell.2022.05.013)
