# Assets

This document provides a comprehensive overview of the supported assets, evaluation tasks, and metrics available on our platform. It is structured to help you quickly navigate the resources and understand their functionalities.

> **Note**: The information provided in this document may not reflect the latest updates. For the most current details, please refer to the [Virtual Cell Platform documentation](#).

---

## Cross-Asset Policies

Cross-asset policies establish unified rules and guidelines that ensure consistency, standardization, and interoperability across all assets. These policies are designed to streamline workflows and maintain quality across diverse applications.

> **TODO**: Expand this section with specific examples and implementation details for cross-asset policies.

---

## Datasets

The platform provides curated datasets to support model training and evaluation. These datasets include:

- **Single-cell RNA-seq data**: High-resolution datasets for transcriptomics analysis.
- **Perturbation datasets**: Data for modeling and predicting cellular responses.

These datasets ensure comprehensive coverage of use cases. For more details, please refer to the [Virtual Cell Platform documentation](#).

---

## Supported Models

Our platform supports a range of state-of-the-art models tailored for transcriptomics and single-cell analysis. These models are optimized for scalability, accuracy, and interpretability. For the latest updates, please refer to the [Virtual Cell Platform documentation](#).

---

## Supported Evaluation Tasks

Evaluation tasks are designed to rigorously assess model performance across various dimensions. Each task includes methods for execution and metric computation, providing a structured approach to benchmarking model capabilities:

- **Clustering Task**: Evaluates clustering quality using techniques like Leiden clustering on embeddings.
- **Embedding Task**: Measures the separation and structure of classes in the embedding space.
- **Label Prediction Task**: Performs cross-validation to predict labels and assess classification accuracy.
- **Integration Task**: Quantifies batch correction effectiveness and cell type preservation.
- **Perturbation Task**: Compares predicted perturbations to ground truth using metrics like MSE and correlation.

These tasks ensure that models are evaluated comprehensively and consistently.

---

## Supported Metrics

Metrics serve as quantitative benchmarks for evaluating model performance. They are grouped by task type for clarity and ease of interpretation, ensuring that evaluations are both rigorous and actionable:

### Clustering
- **Adjusted Rand Index (ARI)**: Measures similarity between clustering results and ground truth.
- **Normalized Mutual Information (NMI)**: Quantifies the amount of shared information between clusters.

### Embedding
- **Silhouette Score**: Assesses the quality of clustering in the embedding space.

### Integration
- **Entropy per Cell**: Evaluates batch correction effectiveness.
- **Batch Silhouette Score**: Measures the separation of batches in the integrated space.

### Perturbation
- **Mean Squared Error (MSE)**: Quantifies prediction accuracy.
- **R-squared (R²)**: Measures the proportion of variance explained by the model.
- **Jaccard Index**: Compares the similarity between predicted and ground truth sets.

### Label Prediction
- **Accuracy**: Measures the proportion of correct predictions.
- **F1 Score**: Balances precision and recall for classification tasks.
- **Precision**: Evaluates the proportion of true positives among predicted positives.
- **Recall**: Measures the proportion of true positives among actual positives.
- **Area Under the Receiver Operating Characteristic Curve (AUROC)**: Assesses the model's ability to distinguish between classes.

These metrics provide actionable insights into model performance, enabling targeted improvements and robust benchmarking.
