# Visualizing Benchmark Results

This guide explains how to visualize and interpret benchmark results stored in JSON or YAML formats. It includes an example of using Python to load and plot the results, as well as generating UMAP visualizations for embeddings.

## Overview

Benchmark results are JSON, and include metrics like Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and others. This guide demonstrates how to process these results for visualization and generate UMAP embeddings for deeper insights.

## Example JSON Output

Here is an example of the JSON structure for benchmark results:

```json
{
    "Example_Model_A": [
        {
            "metric_type": "adjusted_rand_index",
            "value": -4.2435515076099105e-05,
            "params": {}
        },
        {
            "metric_type": "normalized_mutual_info",
            "value": 0.01401707774484544,
            "params": {}
        }
    ]
}
```

## Visualizing Results with Python

You can use Python libraries like `json`, `pandas`, and `matplotlib` to load and visualize the results.

### Example Code for Benchmark Metrics

The following code demonstrates how to load benchmark results from a JSON file and plot a bar chart for a specific metric (e.g., Adjusted Rand Index):

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results from a JSON file
with open("results.json") as f:
    results = json.load(f)

# Flatten the results into a DataFrame
data = []
for model_name, metrics in results.items():
    for metric in metrics:
        data.append({
            "model_name": model_name,
            "metric_type": metric["metric_type"],
            "value": metric["value"]
        })

df = pd.DataFrame(data)

# Filter for a specific metric (e.g., Adjusted Rand Index)
ari_df = df[df['metric_type'] == "adjusted_rand_index"]

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(ari_df['model_name'], ari_df['value'], color='skyblue')
plt.xlabel("Model")
plt.ylabel("Adjusted Rand Index")
plt.title("Clustering Performance (ARI)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Generating UMAP Visualizations

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be used to visualize high-dimensional embeddings. The following code demonstrates how to generate UMAP visualizations for embeddings stored in a dataset object:

```python
import scanpy as sc

# Assuming `dataset` is an object with an AnnData attribute `adata`
# and a method `get_output` to retrieve the embedding
adata = dataset.adata

# Add the embedding to the AnnData object
adata.obsm['X_emb'] = dataset.get_output(...)  # Replace `...` with the appropriate method arguments

# Compute the neighborhood graph using the embedding
sc.pp.neighbors(adata, use_rep='X_emb')

# Compute UMAP
sc.tl.umap(adata)

# Plot the UMAP
sc.pl.umap(adata, color=['batch', 'cell_type'], title="UMAP Visualization")
```

### Tips for Customization

- Replace `color=['batch', 'cell_type']` with other metadata fields available in your AnnData object.
- Adjust UMAP parameters (e.g., `n_neighbors`, `min_dist`) in `sc.pp.neighbors` for different visualization effects.
