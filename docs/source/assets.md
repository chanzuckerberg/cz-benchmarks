# Assets

## Table of Contents

- [Task Descriptions](#task-descriptions)
- [Model Descriptions](#models-descriptions)
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


## Models Descriptions

| Model                                                                          | Description                                                                                                                                                                                                                                            | Link                                                                                              |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| AIDO.Cell 3M                                                                   | Transformer-based foundation model capable of handling the entire human transcriptome as input and demonstrating performance on tasks such as zero-shot clustering, cell type classification, and perturbation modeling.                               | [Model card](https://virtualcellmodels.cziscience.com/model/01964078-54e7-7937-8817-0c53dda9c153) |
| Geneformer  gf-12L-95M-i4096                                                   | A foundation model for single-cell data that generates meaningful embeddings of cells that can then be used for a wide variety of downstream tasks in a zero-shot manner.                                                                              | [Hugging face](https://huggingface.co/ctheodoris/Geneformer)                                      |
| scGenePTGO-all, fine tuned, Adamson                                            | A single-cell model for perturbation prediction. This is a model variation fine-tuned on the gene ontology annotations, molecular function annotations, cellular component annotations, biological processes annotations, and Adamson et al. datasets. | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGenePTGO−all, fine-tuned, Norman                                             | A single-cell model for perturbation prediction. This is a model variation fine-tuned on the gene ontology annotations, molecular function annotations, cellular component annotations,biological processes annotations, and Norman et al. datasets.   | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGenePTGO-C, fine-tuned, Adamson                                              | A single-cell model for perturbation prediction. This is a model variation fine-tuned on gene ontology annotation,  gene cellular component annotations, and Adamson et al. datasets.                                                                  | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGenePTGO-C, fine-tuned, Norman                                               | A single-cell model for perturbation prediction. This is a model variation fine-tuned on gene ontology annotation,  gene cellular component annotations, and Norman et al. datasets.                                                                   | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGenePTNCBI+UniProt, fine-tuned, Adamson                                      | A single-cell model for perturbation prediction. This is a model variation fine-tuned on NCBI Gene Card Summaries, UniProt protein summaries, and Adamson et al. datasets.                                                                             | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGenePTNCBI+UniProt, fine-tuned, Norman                                       | A single-cell model for perturbation prediction. This is a model variation fine-tuned on NCBI Gene Card Summaries, UniProt protein summaries, and Norman et al. datasets.                                                                              | [Model card](https://virtualcellmodels.cziscience.com/model/01936eb7-dba3-7f2e-b71a-463a7b173120) |
| scGPT - whole human                                                            | A foundation model designed to integrate and analyze large-scale single-cell multi-omics data using a generative pre-trained transformer (GPT) architecture.                                                                                           | [Model card](https://virtualcellmodels.cziscience.com/model/0193323f-2875-7858-862c-6903bf667543) |
| scVI - Version: CxG scVI trained on Census 2023-12-15, homo sapiens, 63M cells | Uses autoencoding-variational Bayesian optimization to learn the underlying latent state of gene expression and to approximate the distributions that underlie observed expression values, while accounting for batch effects and limited sensitivity. | [Model card](https://virtualcellmodels.cziscience.com/model/0192c0b9-e574-7f7e-bf7d-167ed1ef2ced) |
| TF-Exemplar                                                                    | A generative model trained on 110 million cells from human and four model organisms that demonstrates zero-shot performance for cell type classification across species.                                                                               | [Model card](https://virtualcellmodels.cziscience.com/model/01966441-339f-77f7-aa06-f67636f865dc) |
| TF-Metazoa                                                                     | A generative model trained on 112 million cells spanning all twelve species, demonstrating zero-shot performance for cell type classification across species.                                                                                          | [Model card](https://virtualcellmodels.cziscience.com/model/01966441-339f-77f7-aa06-f67636f865dc) |
| TF-Sapiens                                                                     | A generative model trained on 57 million human-only cells trained for tasks such as  disease state identification in human cells prediction of cell type specific transcription factors and gene-gene regulatory relationships in humans.              | [Model card](https://virtualcellmodels.cziscience.com/model/01966441-339f-77f7-aa06-f67636f865dc) |
| UCE - 33 layer                                                                 | A  zero-shot foundation model for single-cell biology, representing any cell across species, tissues, and disease states in a fixed embedding space where cell organization emerges without predefined cell types.                                     | [Repo](https://github.com/snap-stanford/UCE)                                                      |
| UCE - 4 layer                                                                  | A zero-shot foundation model for single-cell biology, representing any cell across species, tissues, and disease states in a fixed embedding space where cell organization emerges without predefined cell types.                                      | [Repo](https://github.com/snap-stanford/UCE)                                                      |

  

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

  
The following models were benchmarked using the Tabula Sapiens v2 dataset, per tissue:  
- AIDO.Cell 3M
- Geneformer  gf-12L-95M-i4096
- Linear baseline
- scGenePT
- scGPT
- scVI - Census 2023-12-15
- Transcriptformer Examplar
- Transcriptformer Metazoa
- Transcriptformer  Sapiens
- UCE 33-layer 
- UCE 4 -layer

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

  

The following models were benchmarked using the Tabula Sapiens v2 dataset, per tissue:
- AIDO.Cell 3M
- Geneformer  gf-12L-95M-i4096
- Linear baseline
- scGPT
- scVI - Census 2023-12-15
- Transcriptformer Exemplar
- Transcriptformer Metazoa
- Transcriptformer Sapiens
- UCE 33-layer
- UCE 4-layer
    

### Cross-Species Batch Integration

This task evaluates the model's ability to learn representations that are consistent across different species. There is a forward pass of the data (each species is treated as an individual dataset) through the model. Once embeddings are generated for each species, they are concatenated into a single embedding matrix to enable cross-species comparison. Finally, the concatenated embeddings, along with the corresponding species labels, are used to compute evaluation metrics. 

####  Task: Cross-Species Batch Integration

| Metrics          | Description                                                                                                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Entropy per cell | Measures the average entropy of the batch labels within the local neighborhood of each cell. Implemented [here](https://github.com/chanzuckerberg/cellxgene-census/blob/f44637ba33567400820407f4f7b9984e52966156/tools/models/metrics/run-scib.py#L86). |
| Batch silhouette | A modified silhouette score to measure the extent of batch mixing within biological labels. Described by [Luecken et al](https://www.nature.com/articles/s41592-021-01336-8).                                                                           |

The following models were benchmarked using the Spermatogenesis  dataset, per species:

- Transcriptformer Exemplar
- Transcriptformer Metazoa
- UCE 33-layer
- UCE 4-Layer


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

- The following models were benchmarked using the Adamson et al.  dataset:
    - scGenePTGO-all, fine tuned, Adamson
    - scGenePTGO-C, fine-tuned, Adamson
    - scGenePTNCBI+UniProt, fine-tuned, Adamson
    
- The following models were benchmarked using the Norman et al.  dataset:
    - scGenePTGO−all, fine-tuned, Norman
    - scGenePTGO-C, fine-tuned, Norman
    - scGenePTNCBI+UniProt, fine-tuned, Norman
    

## Guidelines for Included Assets

As cz-benchmarks develops, robust governance policies will be developed to support direct community contribution.

At this stage, the cz-benchmarks project represents an initial prototype and policy and project governance are intended to provide transparency and support the project in its current phase. Initial guidelines are as follows:

- All content (models, tasks, metrics) included in cz-benchmarks currently represents a subset of recommendations from CZI staff.
- Models included within the package have been contributed by CZI, on behalf of model developers. Feedback from model developers is being sourced via direct outreach to these individuals.
- Future versions will incorporate an expanded and refined set of assets. However, not all assets are appropriate for inclusion in a benchmarking platform. Benchmark assets are chosen based on overall quality in relation to comparable reference points, current standards in the research community, and relationship to supported priority benchmark domains as outlined in the roadmap [link]. Formal asset contribution and asset governance policies are in development.
- At this phase, the CZI SciTech team will guide initial decisions, coordinate updates, and ensure that all assets conform to policy requirements (licensing, versioning, etc.) through direct collaboration with working groups, composed of domain-specific experts from the broader scientific community and partners. 
- We value your feedback -- feel free to open a GitHub issue or reach out to us at virtualcellmodels@chanzuckerberg.com.
