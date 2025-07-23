# Visualizing Benchmark Results

This guide explains how to visualize and interpret benchmark results stored in JSON format. It includes an example of using Python to load and plot the results, as well as generating UMAP visualizations for embeddings.

## Overview

Benchmark results are stored as a list of `MetricResult` objects in JSON format. This guide demonstrates how to process these results for visualization and generate UMAP embeddings for deeper insights.

## Example JSON Output

Here is an example of the JSON structure for benchmark results:

```JSON
{
    "Example_Model_A": [
        {
            "metric_type": "adjusted_rand_index",
            "value": 0.85,
            "params": {
                "classifier": "lr"
            }
        },
        {
            "metric_type": "normalized_mutual_info",
            "value": 0.76,
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
        entry = {
            "model_name": model_name,
            "metric_type": metric["metric_type"],
            "value": metric["value"]
        }
        # Add params to the entry for detailed filtering
        entry.update(metric.get("params", {}))
        data.append(entry)

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

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that can be used to visualize high-dimensional embeddings. The following code demonstrates how to generate UMAP visualizations for a given embedding.


```python
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

# Assume `embedding` is your high-dimensional data (e.g., from a model)
# and `metadata_df` is a pandas DataFrame with cell metadata (e.g., cell_type)
# Example dummy data:
n_cells = 500
embedding = np.random.rand(n_cells, 128)
metadata_df = pd.DataFrame({
    'cell_type': np.random.choice(['T-cell', 'B-cell', 'Macrophage'], size=n_cells),
    'batch': np.random.choice(['Batch1', 'Batch2'], size=n_cells)
})


# Create an AnnData object to store the embedding and metadata
adata = ad.AnnData(X=embedding)
adata.obs = metadata_df

# Add the embedding to the AnnData object for visualization
adata.obsm['X_emb'] = embedding

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