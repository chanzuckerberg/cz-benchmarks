# Assets

## Table of Contents

- [Task Descriptions](#task-descriptions)
- [Data Descriptions](#data-descriptions)
- [Task Details](#task-details)
    - [Cell Clustering (in embedding space)](#cell-clustering-in-embedding-space)
    - [Metadata Label Prediction - Cell Type Classification](#metadata-label-prediction-cell-type-classification)
    - [Cross-Species Batch Integration](#cross-species-batch-integration)
    - [Genetic Perturbation Prediction](#genetic-perturbation-prediction)
- [Guidelines for Included Assets](#guidelines-for-included-assets)
    


## Task Descriptions

| Task                                                                                | Description                                                                                                                 |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [Cell clustering](#cell-clustering-in-embedding-space) (in embedding space)         | Cluster cells in embedding space and evaluate against known labels (e.g. cell type)                                         |
| [Cell type classification](#metadata-label-prediction-cell-type-classification)   | Use classifiers to predict cell type from embeddings                                                                        |
| [Cross-Species Batch Integration](#cross-species-batch-integration)                 | Evaluate whether embeddings can align multiple species in a shared space                                                    |
| [Genetic perturbation prediction](#genetic-perturbation-prediction)                 | [In progress, subject to further validation] Compare predicted vs ground-truth expression shifts under genetic perturbation |


## Data Descriptions

| Dataset           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                            | Link                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Tabula sapiens V2 | Part of a reference human cell atlas that includes single-cell transcriptomic data for over 500,000 cells representing 26 tissues sampled from male (n = 2) and female (n = 7) donors. Tissues include: bladder, blood, bone marrow, ear, eye, fat, heart, large intestine, liver, lung, lymph node, mammary, muscle, ovary, prostate, salivary gland, skin, small intestine, spleen, stomach, testis thymus, tongue, trachea uterus, and vasculature. | s3://cz-benchmarks-data/datasets/v1/cell_atlases/Homo_sapiens/Tabula_Sapiens_v2/                   |
| Spermatogenesis   | Includes single-nucleus RNA sequencing (snRNA-seq) data for testes from eleven species, including ten representative mammals and a bird. Species include human, mouse, Rhesus macaque, gorilla, chimpanzee, marmoset, chicken, opossum, and platypus.                                                                                                                                                                                                  | s3://cz-benchmarks-data/datasets/v1/evo_distance/testis/                                           |
| Adamson et al.    | Comprises single-cell RNA sequencing (scRNA-seq) data generated from a multiplexed CRISPR screening platform. It captures transcriptional profiles resulting from targeted genetic perturbations, facilitating the systematic study of the unfolded protein response (UPR) at a single-cell resolution.                                                                                                                                                | [Data card](https://virtualcellmodels.cziscience.com/dataset/01933236-960b-7b1a-bfe3-f3ebc7415076) |
| Norman et al.     | Comprises single-cell RNA sequencing (scRNA-seq) data obtained from Perturb-seq experiments. It captures transcriptional profiles resulting from genetic perturbations, facilitating the study of genetic interactions and cellular state landscapes.                                                                                                                                                                                                  | [Data card](https://virtualcellmodels.cziscience.com/dataset/01933237-1bad-7ead-9619-4730290f2df4) |

  

## Task Details

### Cell Clustering (in embedding space)

This task evaluates how well the model's embedding space separates different cell types. There is a forward pass of the data to produce embeddings. The embeddings are then clustered and compared to known cell type labels. 

#### Task: Cell Clustering (in embedding space)

| Mode            | Metrics                                                                                                                                                                                                                                                                                                                                                    | Metric description                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clustering Task | ARI                                                                                                                                                                                                                                                                                                                                                        | Adjusted Rand Index of biological labels and leiden clusters. Described in [Luecken et al.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html) and implemented in [scib-metrics.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html)                                                                                            |
| NMI             | Normalized Mutual Information of biological labels and leiden clusters. Described in [Luecken et al.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html) and implemented in [scib-metrics.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html) |                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Embedding Task  | Silhouette score                                                                                                                                                                                                                                                                                                                                           | Measures cluster separation based on within-cluster and between-cluster distances to evaluate the quality of clusters with respect to biological labels. Described in [Luecken et al.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html) and implemented in [scib-metrics.](https://scib-metrics.readthedocs.io/en/stable/generated/scib_metrics.nmi_ari_cluster_labels_leiden.html) |

  
### Metadata label prediction - Cell type classification

This task evaluates how well model embeddings capture information relevant to cell identity. This is achieved by a forward pass of the data through each model to retrieve embeddings, and then using the embeddings to train different classifiers, in this case we are using Logistic Regression, KNN, and RandomForest,to predict the cell type. To ensure a reliable evaluation, a 5-fold cross-validation strategy is employed. For each split, the classifier's predictions on the held-out data, along with the true cell type labels, are used to compute a range of classification metrics. The final benchmark output for each metric is the average across the 5 cross-validation folds.

#### Task: Metadata label prediction - Cell type classification

| Metrics   | Description                                                                                                                                                                                                                                                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Macro F1  | Measures the harmonic mean of precision and recall; ( 2*tp ) / ( 2 * tp + fp + fn ) where tp = true positives, fn = false negatives, fp = false positives. Implemented [here](https://github.com/chanzuckerberg/cz-benchmarks/blob/7adf963a1bc7cb858e9d5895be9b8ad11633ecab/src/czbenchmarks/metrics/implementations.py#L102).                         |
| Accuracy  | Proportion of correct predictions over total predictions. Implemented [here](https://github.com/chanzuckerberg/cz-benchmarks/blob/7adf963a1bc7cb858e9d5895be9b8ad11633ecab/src/czbenchmarks/metrics/implementations.py#L94).                                                                                                                           |
| Precision | Measures the proportion of true positive predictions among all positive predictions; tp / (tp + fp) where tp = true positives, fp = false positives. Implemented [here](https://github.com/chanzuckerberg/cz-benchmarks/blob/7adf963a1bc7cb858e9d5895be9b8ad11633ecab/src/czbenchmarks/metrics/implementations.py#L110).                               |
| Recall    | Measures the proportion of actual positive instances that were correctly identified;<br><br>tp / (tp + fn) where tp = true positives, fn = false negatives. Implemented [here](https://github.com/chanzuckerberg/cz-benchmarks/blob/7adf963a1bc7cb858e9d5895be9b8ad11633ecab/src/czbenchmarks/metrics/implementations.py#L118).                        |
| AUROC     | Measures the probability that the model will rank a randomly chosen data point belonging to that category higher than a randomly chosen data point not belonging to that category. Implemented [here](https://github.com/chanzuckerberg/cz-benchmarks/blob/7adf963a1bc7cb858e9d5895be9b8ad11633ecab/src/czbenchmarks/metrics/implementations.py#L126). |

  
### Cross-Species Batch Integration

This task evaluates the model's ability to learn representations that are consistent across different species. There is a forward pass of the data (each species is treated as an individual dataset) through the model. Once embeddings are generated for each species, they are concatenated into a single embedding matrix to enable cross-species comparison. Finally, the concatenated embeddings, along with the corresponding species labels, are used to compute evaluation metrics. 

####  Task: Cross-Species Batch Integration

| Metrics          | Description                                                                                                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Entropy per cell | Measures the average entropy of the batch labels within the local neighborhood of each cell. Implemented [here](https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L86). |
| Batch silhouette | A modified silhouette score to measure the extent of batch mixing within biological labels. Described by [Luecken et al](https://www.nature.com/articles/s41592-021-01336-8).                                                                           |


### Genetic Perturbation Prediction
Warning: This task is still in progress. Results are subject to further validation.

This task evaluates the performance of models fine-tuned to predict cellular responses to genetic perturbations. The process involves applying the fine-tuned model to a test dataset and comparing its predictions with observed ground-truth perturbation profiles. Predicted gene expression profiles after perturbation are generated by running a held-out dataset through the fine-tuned model. These predicted profiles are then compared to ground-truth gene expression profiles for the applied perturbations.

#### Task: Genetic Perturbation Prediction

| Metrics                                     | Description |
| ------------------------------------------- | ----------- |
| MSE - top 20 DE genes                       |             |
| MSE - all genes                             |             |
| Pearson Delta Correlation - top 20 DE genes |             |
| Pearson Delta Correlation - all genes       |             |
| Jaccardian Similarity                       |             |


## Guidelines for Included Assets

As cz-benchmarks develops, robust governance policies will be developed to support direct community contribution.

At this stage, the cz-benchmarks project represents an initial prototype and policy and project governance are intended to provide transparency and support the project in its current phase. Initial guidelines are as follows:

- All content (datasets, tasks, metrics) included in cz-benchmarks currently represents a subset of recommendations from CZI staff.
- Future versions will incorporate an expanded and refined set of assets. However, not all assets are appropriate for inclusion in a benchmarking platform. Benchmark assets are chosen based on overall quality in relation to comparable reference points, current standards in the research community, and relationship to supported priority benchmark domains as outlined in the [roadmap](./roadmap.md). Formal asset contribution and asset governance policies are in development.
- **Note**: TranscriptFormer was developed by the CZI AI team using separate task implementations. The cz-benchmarks task definitions, developed by the CZI SciTech team, were not included as a part of TranscriptFormer training and evaluation.
- At this phase, the CZI SciTech team will guide initial decisions, coordinate updates, and ensure that all assets conform to policy requirements (licensing, versioning, etc.) through direct collaboration with working groups, composed of domain-specific experts from the broader scientific community and partners. 
- We value your feedback -- feel free to open a GitHub issue or reach out to us at virtualcellmodels@chanzuckerberg.com.
