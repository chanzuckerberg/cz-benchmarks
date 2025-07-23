# ==============================================================================
#
# End-to-End Tutorial: Benchmarking an scVI Model with czbenchmarks
#
# This script demonstrates a full workflow for evaluating a single-cell model's
# embedding. It covers:
#   1. Setting up the environment.
#   2. Loading a standard benchmark dataset.
#   3. Running inference with a pre-trained scVI model.
#   4. Evaluating the model's embedding on multiple biological tasks.
#   5. Computing a PCA-based baseline for each task.
#   6. Visualizing the results with comparative plots to clearly assess
#      model performance against the baseline.
#
# ==============================================================================

# --- Step 1: Setup and Imports ---

# Create isolated virtual environment for scVI and czbenchmarks (run once)
# You can uncomment the following lines to set up the environment if needed.
# !python3 -m venv .venv_scvi
# !.venv_scvi/bin/python -m pip install --upgrade pip
# !.venv_scvi/bin/python -m pip install ipykernel numpy pandas matplotlib seaborn scvi-tools czbenchmarks tabulate
# !.venv_scvi/bin/python -m ipykernel install --user --name venv_scvi --display-name "Python (.venv_scvi)"
# print("Virtual environment '.venv_scvi' created, dependencies installed, and kernel registered.")

import logging
import sys
import functools

# Import core czbenchmarks modules for dataset loading and task evaluation
from czbenchmarks.datasets import load_dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
)
from czbenchmarks.tasks.clustering import ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTaskInput
import scanpy as sc

# Import scVI and AnnData for model inference and data handling
import scvi
import anndata as ad

# Import visualization and data manipulation libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import os
import boto3

# Set up basic logging to see the library's output
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
sns.set_theme(style="whitegrid")

# --- Step 2: Load a Dataset ---
# Load the pre-configured `tsv2_prostate` dataset.
# The library handles automatic download, caching, and loading.
print("\n" + "=" * 50)
print("--- Loading Dataset ---\n")
dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")
print("✅  Dataset loaded successfully.")
print(f"Loaded data shape: {dataset.adata.shape}")
print(
    f"Labels available: {dataset.labels.name} with {len(dataset.labels.unique())} unique values."
)


# --- Step 3: Run Model Inference and Get Output ---
# Use a pre-trained scVI model to generate cell embeddings.
print("\n" + "=" * 50)
print("\n--- Running Model Inference ---\n")
# Download model weights from S3 if not already present locally

s3_model_dir = "s3://cz-benchmarks-data/models/v1/scvi_2023_12_15/homo_sapiens/"
local_model_dir = "/tmp/czbenchmarks_scvi_model"

if not os.path.exists(local_model_dir):
    os.makedirs(local_model_dir, exist_ok=True)
    s3 = boto3.resource("s3")
    bucket_name = "cz-benchmarks-data"
    prefix = "models/v1/scvi_2023_12_15/homo_sapiens/"
    bucket = s3.Bucket(bucket_name)
    print("⬇️ Downloading model weights from S3...")
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue
        local_path = os.path.join(local_model_dir, os.path.relpath(obj.key, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bucket.download_file(obj.key, local_path)
    print(f"✅ Downloaded model weights from {s3_model_dir} to {local_model_dir} \n")

model_weights_dir = local_model_dir
batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]

print("🔄 Preparing dataset for model inference...")

adata = dataset.adata.copy()
# Create a 'batch' key for the model by concatenating metadata columns
adata.obs["batch"] = functools.reduce(
    lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
)

# Use the scvi-tools API to map our dataset to the reference model
scvi.model.SCVI.prepare_query_anndata(adata, model_weights_dir)
scvi_model = scvi.model.SCVI.load_query_data(adata, model_weights_dir)
scvi_model.is_trained = True

# Optional: Fine-tune the model on the query data.
# Uncomment code below to run fine tuning
# print("Starting scVI model fine-tuning...")
# scvi_model.train(
#     max_epochs=50,
#     plan_kwargs={"lr": 5e-5},
#     early_stopping=True,
#     early_stopping_patience=10,
# )
# print("Fine-tuning complete.")

# Generate the latent representation (the embedding) from the fine-tuned model
model_output = scvi_model.get_latent_representation()
print(f"Generated scVI embedding with shape: {model_output.shape}")

# --- Step 4: Task-by-Task Evaluation and Visualization ---

# This section evaluates the model and baseline on each task and visualizes the results immediately.
all_results = {}
expression_data = dataset.adata.X

# ==============================================================================
# Task 1: Clustering
# Evaluates how well the embedding separates known cell types.
# Metrics: Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
# Higher scores are better.
# ==============================================================================
print("\n" + "=" * 50)
print("\n--- Task 1: Evaluating Clustering Performance ---\n")

# 1. Initialize the task and define inputs
clustering_task = ClusteringTask()
clustering_task_input = ClusteringTaskInput(
    obs=dataset.adata.obs,
    input_labels=dataset.labels,
)

# 2. Run task on model output
clustering_results_model = clustering_task.run(
    cell_representation=model_output,
    task_input=clustering_task_input,
)

# 3. Compute and run baseline
clustering_baseline_embedding = clustering_task.compute_baseline(expression_data)
clustering_results_baseline = clustering_task.run(
    cell_representation=clustering_baseline_embedding,
    task_input=clustering_task_input,
)

# Store results
all_results["clustering"] = {
    "model": [r.model_dump() for r in clustering_results_model],
    "baseline": [r.model_dump() for r in clustering_results_baseline],
}

# 4. Visualize Clustering Results

# a) Bar Chart for Metrics
df_clustering_model = pd.DataFrame(all_results["clustering"]["model"])
df_clustering_baseline = pd.DataFrame(all_results["clustering"]["baseline"])
df_clustering_model["source"] = "scVI Model"
df_clustering_baseline["source"] = "PCA Baseline"
df_clustering = pd.concat([df_clustering_model, df_clustering_baseline])
df_clustering["metric_name"] = df_clustering["metric_type"].apply(lambda x: x.name)


plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_clustering, x="metric_name", y="value", hue="source", palette="viridis"
)
plt.title("Clustering Performance: scVI Model vs. PCA Baseline", fontsize=16)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Metric", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Source")
plt.tight_layout()
plt.savefig("clustering_performance_barplot.png", dpi=150)
plt.close()


# b) UMAP Visualization
def plot_umap(embedding, title, labels):
    adata_vis = ad.AnnData(embedding)
    adata_vis.obs["cell_type"] = labels.values
    scvi.model.SCVI.setup_anndata(adata_vis)  # Use scvi setup for consistency
    sc.pp.neighbors(adata_vis, use_rep="X")
    sc.tl.umap(adata_vis)
    sc.pl.umap(adata_vis, color="cell_type", title=title, frameon=False, show=False)
    return plt.gcf()


print("Generating UMAP visualizations...")
fig1 = plot_umap(model_output, "UMAP of scVI Embedding", dataset.labels)
fig2 = plot_umap(
    clustering_baseline_embedding, "UMAP of PCA Baseline Embedding", dataset.labels
)
fig1.savefig("umap_scvi_embedding.png", dpi=150)
fig2.savefig("umap_pca_baseline_embedding.png", dpi=150)
plt.close(fig1)
plt.close(fig2)

# ==============================================================================
# Task 2: Embedding Quality
# Evaluates the intrinsic quality of the embedding space using Silhouette Score.
# Higher scores indicate better-defined clusters.
# ==============================================================================
print("\n" + "=" * 50)
print("\n--- Task 2: Evaluating Embedding Quality ---\n")

# 1. Initialize task and define inputs
embedding_task = EmbeddingTask()
embedding_task_input = EmbeddingTaskInput(input_labels=dataset.labels)

# 2. Run task on model output
embedding_results_model = embedding_task.run(model_output, embedding_task_input)

# 3. Compute and run baseline
embedding_baseline_embedding = embedding_task.compute_baseline(expression_data)
embedding_results_baseline = embedding_task.run(
    embedding_baseline_embedding, embedding_task_input
)

# Store results
all_results["embedding"] = {
    "model": [r.model_dump() for r in embedding_results_model],
    "baseline": [r.model_dump() for r in embedding_results_baseline],
}

# 4. Visualize Embedding Results
df_embedding_model = pd.DataFrame(all_results["embedding"]["model"])
df_embedding_baseline = pd.DataFrame(all_results["embedding"]["baseline"])
df_embedding_model["source"] = "scVI Model"
df_embedding_baseline["source"] = "PCA Baseline"
df_embedding = pd.concat([df_embedding_model, df_embedding_baseline])
df_embedding["metric_name"] = df_embedding["metric_type"].apply(lambda x: x.name)


plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_embedding, x="metric_name", y="value", hue="source", palette="plasma"
)
plt.title("Embedding Quality: scVI Model vs. PCA Baseline", fontsize=16)
plt.ylabel("Silhouette Score", fontsize=12)
plt.xlabel("")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig("embedding_quality_barplot.png", dpi=150)
plt.close()

# ==============================================================================
# Task 3: Metadata Label Prediction
# Evaluates how well the embedding can be used to predict cell types.
# Metrics: Accuracy, F1 Score, AUROC, etc., averaged over 5 folds.
# Higher scores are better.
# ==============================================================================
print("\n" + "=" * 50)
print("\n--- Task 3: Evaluating Metadata Label Prediction ---\n")

# 1. Initialize task and define inputs
prediction_task = MetadataLabelPredictionTask()
prediction_task_input = MetadataLabelPredictionTaskInput(labels=dataset.labels)

# 2. Run task on model output
prediction_results_model = prediction_task.run(model_output, prediction_task_input)

# 3. Compute and run baseline
prediction_baseline_embedding = prediction_task.compute_baseline(expression_data)
prediction_results_baseline = prediction_task.run(
    prediction_baseline_embedding, prediction_task_input
)

# Store results
all_results["prediction"] = {
    "model": [r.model_dump() for r in prediction_results_model],
    "baseline": [r.model_dump() for r in prediction_results_baseline],
}

# 4. Visualize Prediction Results
df_pred_model = pd.DataFrame(all_results["prediction"]["model"])
df_pred_baseline = pd.DataFrame(all_results["prediction"]["baseline"])
df_pred_model["source"] = "scVI Model"
df_pred_baseline["source"] = "PCA Baseline"
df_pred = pd.concat([df_pred_model, df_pred_baseline])
df_pred["metric_name"] = df_pred["metric_type"].apply(lambda x: x.name)
df_pred["classifier"] = df_pred["params"].apply(
    lambda p: p.get("classifier", "Overall")
)

# Filter for overall metrics for a cleaner plot
df_pred_overall = df_pred[df_pred["classifier"].str.contains("MEAN")]

plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_pred_overall, x="metric_name", y="value", hue="source", palette="magma"
)
plt.title(
    "Metadata Label Prediction Performance: scVI Model vs. PCA Baseline", fontsize=16
)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Metric", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig("metadata_label_prediction_barplot.png", dpi=150)
plt.close()

# --- Step 5: Final Summary ---
# Print all results in a single JSON object for programmatic access.
print("\n--- Final Consolidated Results (JSON) ---")
# print(json.dumps(all_results, indent=2, default=str))

# Pretty print all_results as a formatted table


def flatten_results(results_dict):
    rows = []
    for task, sources in results_dict.items():
        for source, metrics in sources.items():
            for metric in metrics:
                row = {
                    "Task": task.capitalize(),
                    "Source": "scVI Model" if source == "model" else "PCA Baseline",
                    "Metric": metric.get("metric_type", ""),
                    "Value": metric.get("value", ""),
                    "Params": metric.get("params", ""),
                }
                rows.append(row)
    return rows


try:
    rows = flatten_results(all_results)
    print("\n--- All Results (Tabular Format) ---")
    print(tabulate(rows, headers="keys", tablefmt="github", showindex=False))
except ImportError:
    print("Install 'tabulate' to see results as a table: pip install tabulate")
