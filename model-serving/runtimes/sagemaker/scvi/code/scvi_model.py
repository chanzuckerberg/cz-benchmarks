import functools
import os
import yaml
from pathlib import Path
import scvi
from anndata import AnnData
import anndata as ad
import pandas as pd
import scipy.sparse as sparse
import logging
from utils import download_from_s3
logger = logging.getLogger(__name__)


class SCVI:
    """
    This class loads scvi-tools models and performs inference.
    It expects a dictionary of artifact paths in self.artifacts.
    """

    def __init__(self, artifacts: dict = None):
        # Initialize the SCVI model. If artifacts are not provided, load them from config.yaml
        if artifacts is None:
            config_path = os.environ.get("SCVI_CONFIG", "config.yaml")
            with open(config_path, "r") as f:
                artifacts = yaml.safe_load(f)
        self.artifacts = artifacts

    def predict(self, adata: AnnData, hvg_file_path: str, reference_model_path: Path):
        batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]

        # Filter input anndata by HVGs
        logger.info(f"Filtering adata by HVGs")
        adata = self._filter_adata_by_hvg(adata, hvg_file_path)

        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )

        logger.info(f"Preparing query anndata")
        scvi.model.SCVI.prepare_query_anndata(
            adata, str(reference_model_path), return_reference_var_names=True
        )
        logger.info(f"Loading query data")
        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(reference_model_path),
        )
        vae_q.is_trained = True

        # Get latent representation
        logger.info(f"Getting latent representation")
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)
        logger.info(f"Latent representation shape: {qz_m.shape}")
        return qz_m

    def _filter_adata_by_hvg(self, adata: ad.AnnData, hvg_file_path: str) -> ad.AnnData:
        """Filter adata by HVGs for the specified organism, downloading HVG names if not already present."""

        if not os.path.exists(hvg_file_path):
            raise FileNotFoundError(
                f"HVG file could not be found at {hvg_file_path}"
            )

        adata = adata.copy()

        hvg = pd.read_csv(hvg_file_path)
        adata.var["feature_id"] = adata.var["feature_id"].astype(str)
        hvg["feature_id"] = hvg["feature_id"].astype(str)

        # Remove duplicate columns if present
        adata.var = adata.var.loc[:, ~adata.var.columns.duplicated()]

        mask = adata.var["feature_id"].isin(hvg["feature_id"])
        adata_filtered = adata[:, mask].copy()

        missing_features = set(hvg.feature_id) - set(adata.var.feature_id)
        if missing_features:
            logger.info(
                f"WARNING: {len(missing_features)} HVGs are not present in the AnnData object"
            )
            missing_var = pd.DataFrame({"feature_id": list(missing_features)})
            missing_var["feature_name"] = missing_var["feature_id"]
            missing_var.set_index("feature_name", inplace=True)
            missing_X = sparse.csr_matrix((adata.n_obs, len(missing_features)))
            adata_missing = ad.AnnData(
                X=missing_X, var=missing_var, obs=adata_filtered.obs.copy()
            )
            adata_concat = ad.concat(
                [adata_filtered, adata_missing], axis=1, join="outer", merge="first"
            )
        else:
            adata_concat = adata_filtered

        hvg_unique = hvg.drop_duplicates(subset="feature_id")
        adata_reordered = ad.AnnData(
            X=adata_concat[:, hvg_unique.feature_id].X,
            obs=adata_concat.obs.copy(),
            var=adata_concat.var.loc[hvg_unique.feature_id].copy(),
        )
        return adata_reordered

    def _download_hvg_names(self, organism: str):
        """
        Downloads HVG names from S3 for the specified organism.

        Args:
            organism (str): The organism identifier (e.g., 'homo_sapiens').
        """
        hvg_val = self.artifacts.get(organism, {}).get("hvg_names")
        if hvg_val and isinstance(hvg_val, str) and hvg_val.startswith("s3://"):
            local_dir = "/tmp"
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(hvg_val))
            if not os.path.exists(local_path):
                download_from_s3(hvg_val, local_path)
            self.artifacts[f"hvg_names_{organism}"] = local_path
            self.artifacts[organism]["hvg_names"] = local_path
            return local_path

    def _download_model_weights(self, organism: str):
        """
        Downloads model weights from S3 for the specified organism.

        Args:
            organism (str): The organism identifier (e.g., 'homo_sapiens').
        """
        mw_val = self.artifacts.get(organism, {}).get("model_weights")
        if mw_val and isinstance(mw_val, str) and mw_val.startswith("s3://"):
            local_dir = "/tmp"
            local_path = os.path.join(local_dir, os.path.basename(mw_val))
            if not os.path.exists(local_path):
                download_from_s3(mw_val, local_path)
            self.artifacts[f"model_weights_{organism}"] = local_path
            self.artifacts[organism]["model_weights"] = local_path
            return local_path
