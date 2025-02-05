import scanpy as sc
import pandas as pd

file_path = 'tests/datasets/Homo_sapiens_ERP132584.h5ad'
adata = sc.read_h5ad(file_path)

print('Summary:\n=======\n')
print(adata)

print("\nObservations (cells): Key: obs\n======================\n")
print(adata.obs.keys())
print(adata.obs.head())

print("\nVariables (genes): Key: var \n==================\n")
print(adata.var.keys())
print(adata.var.head())


print("\nLayers:\n===========\n")
print(adata.layers.keys())
print(adata.layers)


print("\nX matrix (cell by gene matrix) Key: X.shape:\n===============================\n")
print(adata.X.shape)


cell_by_gene_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
print("\nCell by Gene DataFrame: key: X\n=======================\n")
print(cell_by_gene_df.shape)
print(cell_by_gene_df.head())


print("\nMetadata: Key: obs.keys \n=======================\n")
print(adata.obs.keys())


print("\nUnstructured: Key: uns.keys \n=======================\n")
print(adata.uns.keys())
