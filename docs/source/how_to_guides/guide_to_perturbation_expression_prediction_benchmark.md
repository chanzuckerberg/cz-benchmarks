# Guide to Perturbation Expression Prediction Benchmark

## Overview

This task evaluates a model's ability to predict expression for masked genes using the remaining (unmasked) genes for a given cell as context. Genes are randomly selected for masking from the set of genes that are identified as differentially expressed, based on threshold parameters explained in [Masking Parameters](#masking-parameters). For Replogle K562 Essentials[^replogle-k562-essentials], the provided controls used for the differential expression (DE) analysis have been determined based on GEM group and UMI count and are stored in the dataset along with the DE results.

- Datasets contain perturbed and control cells. Matched controls have been determined for each condition and are stored in the unstructured portion of the AnnData under the key `control_cell_ids`. 
- The differential expression results are also stored in the unstructured portion of the AnnData in the keys `de_results_wilcoxon` for analysis that utilized the Wilcoxon signed-rank test. This is the only metric that is currently supported. 
- The gene expression values have not been further processed and should be processed as required for the respective model during inference.

## Dataset Functionality and Parameters

The data loading method accomplishes the following:

- Perturbed cells and their matched controls are concatenated into an AnnData object for each condition. Each condition is then added to the AnnData `obs` metadata column defined by the parameter ``{condition_key}``.
- In the matched data, the perturbations are labeled as ``{perturb}``, and matched control cells are labeled as ``{control_name}_{perturb}``, where ``{perturb}`` is the name of the perturbation and ``{control_name}`` is a configurable parameter.
- Combinatorial perturbations are not currently supported.
- For each condition, a subset of DE genes are sampled from the DE results and masked from their default values. These become the prediction targets for the model.
- The objective is to predict the masked expression values for the prediction targets per cell/condition.

### Data Parameters

The following parameters are used in loading the data:

- `condition_key`: The name of the column in `adata.obs` and the DE results containing condition labels for perturbations and controls. Default is "condition".
- `control_name`: The name used to denote control samples and to form control labels ``{control_name}_{perturb}``. Default is "non-targeting".
- `de_gene_col`: The name of the column in the DE results indicating gene identifiers to be considered for masking. Default is "gene".
- `de_results_path`: CSV path for external DE results. If not provided, DE results are read from the dataset.
  

### Masking Parameters

The following parameters control masking of the DE genes:

- `percent_genes_to_mask`: The fraction of DE genes per condition to mask as prediction targets for the model. Default value is 0.5.
- `min_de_genes_to_mask`: Minimum number of sampled DE genes required for a condition to be eligible for masking. This threshold is applied after the genes are sampled. Default value is 5.
- `pval_threshold`: Maximum adjusted p-value for DE filtering based on the output of the DE analysis. This data must be in the column `pval_adj`. Default value is 1e-4.
- `min_logfoldchange`: Minimum absolute log-fold change to determine when a gene is considered differentially expressed. This data must be in the column `logfoldchange`. Only used when the DE analysis uses Wilcoxon rank-sum. Default value is 1.
- `condition_key`: Key for the column in `adata.obs` specifying conditions (default value is "condition")
- `control_name`: Name of the control condition (default value is "non-targeting")

Parameters shared with other single-cell datasets (e.g., `path`, `organism`, `task_inputs_dir`) are also required but not listed here.

### Saving the Dataset
The outputs of the dataset processing can be saved with 
```python
task_inputs_dir = dataset.store_task_inputs()
```


## Task Functionality and Parameters 

This task evaluates perturbation-induced expression predictions against their ground truth values by calculating predicted and experimental log fold change (LFC) values. The class also calculates a baseline prediction (`compute_baseline` method), which takes as input a `baseline_type`, either `median` (default) or `mean`, that calculates the median or mean expression values, respectively, across all cells in the dataset.

The following parameters are used by the task, via the `PerturbationExpressionPredictionTaskInput` class:  

- `de_results`: DE results used by the dataset class (`SingleCellPerturbationDataset`).
- `adata`: The complete AnnData object containing control-matched and masked data.
- `cell_index`: Sequence of cell barcodes vertically aligned with `cell_representation` matrix.
- `gene_index`: Sequence of gene names horizontally aligned with `cell_representation` matrix.
- `target_conditions_dict`: Dictionary of target conditions whose genes were randomly selected for masking.

If the dataset results were saved as above, they can be loaded with 
```python
task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)
```

### Notes on loading model predictions. 
When a user loads in model predictions, the genes that are predicted on should be a subset of the genes in the replogle dataset, and the cells that are predicted on should be a subset of the cells. At the start of the task, there is a valiation step that ensures that these criteria are met. 

To ensure that the correct values are predicted on, in the PerturbationExpressionPredictionTaskInput set the gene_index and cell_index to the genes and cells that were predicted on.  

## Metrics

The task produces per-condition metrics by comparing predicted and ground-truth log fold change (LFC) values for the masked genes:

- **Spearman correlation (rank)**: Rank correlation between the raw predicted and ground truth LFC values.
- **Accuracy (binarized)**: Accuracy after converting continuous LFC values into binary labels (up-regulated or down-regulated).
- **Precision (binarized)**: Fraction of predicted positives that are true positives of the binarized labels.
- **Recall (binarized)**: Fraction of true positives that are correctly identified of the binarized labels.
- **F1 score (binarized)**: Harmonic mean of precision and recall of the binarized labels.

Results are generated for each perturbation condition separately. Downstream reporting may aggregate scores across conditions (e.g., mean and standard deviation).

[^replogle-k562-essentials]: Replogle, J. M., Elgamal, R. M., Abbas, A. et al. Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. Cell, 185(14):2559–2575.e28 (2022). [DOI](https://doi.org/10.1016/j.cell.2022.05.013)
