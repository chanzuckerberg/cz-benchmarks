# Assets

## Cross-asset policies
## Models
## Datasets
## Tasks
## Metrics


## Supported Models

| Model      | Description                                   |
|------------|-----------------------------------------------|
| SCVI       | Variational inference for single-cell RNA-seq |
| SCGPT      | GPT for transcriptomics                       |
| Geneformer | Transformer model for gene expression         |
| scGenePT   | Gene perturbation prediction transformer      |
| UCE        | Universal Cell Embedding model                |



## Supported Evaluation Tasks

Each task implements `_run_task()` and `_compute_metrics()`

-   **ClusteringTask**: Uses Leiden clustering on embeddings
-   **EmbeddingTask**: Evaluates separation of classes
-   **LabelPredictionTask**: Cross-validation-based prediction
-   **IntegrationTask**: Batch correction and cell type preservation
-   **PerturbationTask**: MSE and correlation on predicted vs. ground truth perturbations
    

## Supported Metrics

Metrics are organized by tags:

-   `clustering`: ARI, NMI
-   `embedding`: Silhouette score
-   `integration`: Entropy per cell, Batch silhouette
-   `perturbation`: MSE, R2, Jaccard
-   `label_prediction`: Accuracy, F1, Precision, Recall, AUROC