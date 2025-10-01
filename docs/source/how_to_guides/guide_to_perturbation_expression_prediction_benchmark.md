# Guide to Perturbation Expression Prediction Benchmark

## Overview

This task evaluates a model's ability to predict expression for masked genes using the remaining (unmasked) genes for a given cell as context. Genes are randomly selected for masking from the set of genes that are identified as differentially expressed, based on threshold parameters explained in [Masking Parameters](#masking-parameters). For Replogle K562 Essentials[^replogle-k562-essentials], the provided controls used for the differential expression (DE) analysis have been determined based on GEM group and UMI count and are stored in the dataset along with the DE results.

- Single cell perturbation datasets contain perturbed and control cells. Matched controls have been determined for each condition and are stored in the unstructured portion of the AnnData under the key `control_cell_map`.
- The differential expression results are also stored in the unstructured portion of the AnnData in the key `de_results_wilcoxon`. This analysis utilized the Wilcoxon rank-sum test, and is currently the only DE selection method that is supported, but additional options are planned. 
- The gene expression values in the dataset are counts, provided in unmodified form relative to that from the original authors. Additional preprocessing, such as scaling and log transformation, should be performed by the user.

This benchmark is designed for evaluation by any model that produces a prediction matrix whose cells (rows) and genes (columns) can be aligned with those used by the dataset. The task ensures alignment by validating gene and cell indices against the dataset. The predictions provided to the task can be in any unit (e.g. counts, log transformed) that is monotonic to the differential expression results (log2FC).

## Dataset Functionality and Parameters

The data loading method accomplishes the following:

- Perturbed cells and their matched controls are selected and indexed to create a new AnnData object for each condition. Conditions are stored in AnnData `obs` metadata column defined by the parameter ``{condition_key}``.
- In the control matched data, the perturbations are labeled as ``{perturb}``, and matched control cells are labeled as ``{control_name}_{perturb}``, where ``{perturb}`` is the name of the perturbation and ``{control_name}`` is a configurable parameter.
- Combinatorial perturbations are not currently supported.
- For each condition, a subset of DE genes are sampled and their default values are masked. These become the prediction targets for the model.
- The objective is for the model to predict the masked expression values for the prediction targets per cell and per condition.

### Data Parameters

The following parameters are used in loading the data:

- `condition_key`: The name of the column in `adata.obs` and in the DE results containing condition labels for perturbations and controls. Default is "condition".
- `control_name`: The name used to denote control samples and to form control labels ``{control_name}_{perturb}``. Default is "ctrl".
- `de_gene_col`: The name of the column in the DE results indicating gene identifiers to be considered for masking. Default is "gene_id".
- `de_metric_col`: The name of the metric column in the differential expression data. Default is "logfoldchange".
- `de_pval_col`: The name of the p-value column in the differential expression data. Default is "de_pval_col".

### Masking Parameters

The following parameters control masking of the DE genes:

- `percent_genes_to_mask`: The fraction of DE genes per condition to mask as prediction targets for the model. Default value is 0.5.
- `min_de_genes_to_mask`: Minimum number of sampled DE genes required for a condition to be eligible for masking. This threshold is applied after the genes are sampled. Default value is 5.
- `pval_threshold`: Maximum adjusted p-value for DE filtering based on the output of the DE analysis. This data must be in the column `pval_adj`. Default value is 1e-4.
- `min_logfoldchange`: Minimum absolute log-fold change to determine when a gene is considered differentially expressed. This data must be in the column `logfoldchange`. Only used when the DE analysis uses Wilcoxon rank-sum. Default value is 1.
- `target_conditions_override`: An externally supplied list of target conditions for customized masking. This overrides the default sampling of genes for masking in `target_conditions_dict`. 

The parameters `condition_key` and `control_name` are as described above and used for masking. Parameters shared with other single-cell datasets (e.g., `path`, `organism`, `task_inputs_dir`, `random_seed`) are also required but not described here.

### Saving the Dataset

To cache and reuse dataset outputs without re-running preprocessing, the outputs of the dataset can be saved with:

  ```python
  task_inputs_dir = dataset.store_task_inputs()
  ```

## Task Functionality and Parameters 

This task evaluates predictions of perturbation-induced changes in gene expression against their ground truth values by correlating their values. The predictions provided to the task can be in any format that is monotonic with the differential expression results. Predicted changes are computed per condition as the difference in mean expression between perturbed and matched control cells, for the subset of masked genes.

The task class also calculates a baseline prediction (`compute_baseline` method), which takes as input a `baseline_type`, either `median` (default) or `mean`, that calculates the median or mean expression values, respectively, across all masked values in the dataset.

The following parameters are used by the task input class, via the [`PerturbationExpressionPredictionTaskInput`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html) class:  

- `adata`: The AnnData object produced when the data are loaded by the dataset class ([`SingleCellPerturbationDataset`](../autoapi/czbenchmarks/datasets/single_cell_perturbation/index.html)), containing control-matched and masked data.
- `pred_effect_operation`: This determines how to compute the effect of between treated and control mean predictions. There are two possible values: "difference" uses `mean(treated) - mean(control)` and is generally safe across scales; "ratio" uses `log((mean(treated)+eps)/(mean(control)+eps))` when means are all positive. The default is "ratio".
- `cell_index`: Sequence of user-provided cell is vertically aligned with `cell_representation` matrix, which contains the predictions from the model.
- `gene_index`: Sequence of user-provided gene names horizontally aligned with `cell_representation` matrix, which contains the predictions from the model.

The main task, [`PerturbationExpressionPredictionTask`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html) requires only an optional random seed as input. The dataclass ([`PerturbationExpressionPredictionTaskInput`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html)) and a matrix of model predictions is required to be provided to the `run` method which executes the task.

The task returns a dataclass, [`PerturbationExpressionPredictionOutput`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html), which contains the following:

- `pred_mean_change_dict`: The predicted fold change for the masked genes based on the model.
- `true_mean_change_dict`: The ground truth fold change based on the differential expression results provided by the dataset.

These outputs are then provided to the metric for computation of the [Spearman correlation](../autoapi/czbenchmarks/metrics/implementations/index.html).


### Notes on Loading Model Predictions

When a user loads in model predictions, the cells and genes whose expression values are predicted should each be a subset of those in the dataset. At the start of the task, validation is performed to ensure these criteria are met. 

It is essential that the mapping of the cells (rows) and genes (columns) from the model expression predictions to those in the dataset is correct. Thus, the [`PerturbationExpressionPredictionTaskInput`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html) requires a `gene_index` and `cell_index` to be provided by the user for validation.

If the user has an AnnData (model_adata) with model predictions, and a [`SingleCellPerturbationDataset`]() with loaded data, [`PerturbationExpressionPredictionTaskInput`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html) can be prepared using the [`build_task_input_from_predictions`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index.html) function:

  ```python
  task_input = build_task_input_from_predictions(predictions_adata=model_adata, dataset_adata=dataset.adata)
  ```

## Metrics

The task produces a per-condition correlation by comparing predicted and ground-truth expression values for the masked genes. The comparison metric is:

- **Spearman correlation (rank)**: Rank correlation between the raw predicted and ground truth values. As this is a rank metric, the predictions can be supplied in any units that are monotonic to the ground truth data, which uses log fold change (Log2FC).


Results are generated for each perturbation condition separately. Downstream reporting may aggregate scores across conditions (e.g., mean and standard deviation).

For large-scale benchmarks, metrics can be exported to CSV/JSON via the provided [`czbenchmarks.tasks.utils.print_metrics_summary helper`](../autoapi/czbenchmarks/tasks/utils/index.html), or integrated into custom logging frameworks.

## Example Usage

For example use cases, see the example script `examples/example_perturbation_expression_prediction.py`. 

[^replogle-k562-essentials]: Replogle, J. M., Elgamal, R. M., Abbas, A. et al. Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. Cell, 185(14):2559–2575.e28 (2022). [DOI](https://doi.org/10.1016/j.cell.2022.05.013)
