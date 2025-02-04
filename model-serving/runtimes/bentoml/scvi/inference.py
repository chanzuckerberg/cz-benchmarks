# -----------------------------
# File: inference.py
# -----------------------------
import functools
from anndata import AnnData, read_h5ad
import scvi
import anndata as ad
import pandas as pd
from scipy import sparse

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SCVI:
    """
    This class loads scvi-tools models and performs inference. 
    It expects a dictionary of artifact paths in self.artifacts.
    """

    def __init__(self, artifacts: dict):
        # Make sure SCVI class has an instance variable called `artifacts`.
        self.artifacts = artifacts

    def predict(self, input_file, params={"organism": "homo_sapiens"}):
        """
        Perform inference by reading an h5ad file from `input_file`,
        filtering by HVGs, and running the SCVI model loaded from disk.
        """
        adata = read_h5ad(input_file)
        hvg_file = self.artifacts[f"hvg_names_{params['organism']}"]

        # Use the full path when loading the model weights
        model_dir = Path(self.artifacts[f"model_weights_{params['organism']}"]).absolute().parent
        return SCVI._predict(adata, hvg_file, model_dir)

    @staticmethod
    def _predict(adata: AnnData, hvg_file: str, model_dir: Path):
        batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]

        # Filter input anndata by HVGs
        adata = SCVI._filter_adata_by_hvg(adata, hvg_file)
        
        # Combine batch keys into a single "batch" column
        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )

        # Prepare a query anndata
        scvi.model.SCVI.prepare_query_anndata(
            adata, str(model_dir), return_reference_var_names=True
        )

        # Load query data and set "is_trained" to True
        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(model_dir),
        )
        vae_q.is_trained = True

        # Get latent representation
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)
        return qz_m

    @staticmethod
    def _filter_adata_by_hvg(adata: ad.AnnData, hvg_path: str) -> ad.AnnData:
        """Filter adata by HVGs."""
        # Create a full copy at the start to avoid view issues
        adata = adata.copy()
        
        hvg = pd.read_csv(hvg_path)
        adata.var["feature_id"] = adata.var["feature_id"].astype(str)
        hvg["feature_id"] = hvg["feature_id"].astype(str)

        # Filter adata by only genes that are present in hvg.feature_id
        # FIXME: Drop duplicate "feature_id" columns, only needed to fix example.h5ad! This should be removed after fixing the h5ad itself.
        adata.var = adata.var.loc[:, ~adata.var.columns.duplicated()]
        
        mask = adata.var["feature_id"].isin(hvg["feature_id"])
        adata_filtered = adata[:, mask].copy()

        # Check which features are missing from hvg.feature_id
        missing_features = set(hvg.feature_id) - set(adata.var.feature_id)

        if missing_features:
            logger.info(f"WARNING:{len(missing_features)} HVGs are not present in the AnnData object")
            # Create an empty adata with missing genes as all zeros, make the array sparse
            missing_var = pd.DataFrame({"feature_id": list(missing_features)})
            missing_var["feature_name"] = missing_var["feature_id"]
            missing_var.set_index("feature_name", inplace=True)
            missing_X = sparse.csr_matrix((adata.n_obs, len(missing_features)))
            # Initialize AnnData with minimal metadata
            adata_missing = ad.AnnData(
                X=missing_X,
                var=missing_var,
                obs=adata_filtered.obs.copy()  # Make sure to copy the obs
            )

            # Concatenate the filtered adata with the missing genes adata
            adata_concat = ad.concat(
                [adata_filtered, adata_missing],
                axis=1,
                join="outer",
                merge="first"
            )
        else:
            adata_concat = adata_filtered

        # Reorder the adata based on hvg.feature_id
        
        # Ensure there are no duplicates in hvg.feature_id
        hvg_unique = hvg.drop_duplicates(subset="feature_id")

        # Create a new AnnData object instead of a view
        adata_reordered = ad.AnnData(
            X=adata_concat[:, hvg_unique.feature_id].X,
            obs=adata_concat.obs.copy(),
            var=adata_concat.var.loc[hvg_unique.feature_id].copy()
        )
        return adata_reordered
