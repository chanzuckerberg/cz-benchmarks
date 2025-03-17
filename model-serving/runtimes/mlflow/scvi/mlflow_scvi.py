import functools
import os
from anndata import AnnData, read_h5ad
import mlflow
import mlflow.pyfunc
import scvi
import anndata as ad
import pandas as pd
from scipy import sparse

import logging
from pathlib import Path
from databricks.sdk import WorkspaceClient
import requests
from urllib.request import urlretrieve
import boto3

logger = logging.getLogger(__name__)


class MLflowSCVI(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Create a temporary directory for storing model files, organized by organism
        # MLflow artifacts must be a flat directory, but the scVI model specifically requires a file called model.pt, 
        # so the organism-specific model.pt files must be in separate directory
        self.artifact_root_dir = Path("/tmp/mlflow_scvi/")
        for organism in ["homo_sapiens", "mus_musculus"]:
            model_organism_dir = (self.artifact_root_dir / organism)
            model_organism_dir.mkdir(parents=True, exist_ok=True)

            model_file = context.artifacts[f"model_weights_{organism}"]
            symlink_path = model_organism_dir / "model.pt"
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(model_file)
            print(f"Model file: {model_file} symlinked to {symlink_path}")

            hvg_names_file = context.artifacts[f"hvg_names_{organism}"]
            symlink_path = model_organism_dir / "hvg_names.csv.gz"
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(hvg_names_file)
            print(f"HVG Names file: {hvg_names_file} symlinked to {symlink_path}")



    def predict(self, context, model_input, params={"organism": "homo_sapiens"}):
        """MLflow prediction entrypoint. Handles input and config."""

        print(context.artifacts)
        print(model_input)
        print(params)
        
        # adata = pickle.loads(model_input.iloc[0, 0])
        adata_url = str(model_input.iloc[0, 0])
        print(adata_url)
        
        if adata_url.startswith("dbfs:/"):
            # FIXME: This does not work (locally, at least). dbfs.copy() simply does not write to /tmp!?
            if not all(env_var in os.environ for env_var in ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]):
                raise EnvironmentError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set as environment variables")
            client = WorkspaceClient()
            local_path = "/tmp/input_adata.h5ad"
            client.dbfs.copy(adata_url, local_path, overwrite=True)
            print(f"Downloaded input data file from {adata_url}")
        elif adata_url.startswith("s3://"):
            s3 = boto3.client('s3')
            bucket_name = adata_url.split('/')[2]
            key = '/'.join(adata_url.split('/')[3:])
            local_path = "/tmp/input_data.h5ad"

            s3.download_file(bucket_name, key, local_path)
            print(f"Downloaded input data file from {adata_url}")
        elif adata_url.startswith("https://") or adata_url.startswith("http://"):
            local_path = "/tmp/input_data.h5ad"
            urlretrieve(adata_url, local_path)
            response = requests.get(adata_url)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded input data file from {adata_url}")
        else:
            local_path = adata_url
            
        adata = read_h5ad(local_path)

        model_dir = self.artifact_root_dir / params["organism"]
        hvg_file = self.artifact_root_dir / params["organism"] / "hvg_names.csv.gz"
        return MLflowSCVI._predict(adata, hvg_file, model_dir)


    @staticmethod
    def _predict(adata: AnnData, hvg_file: str, model_dir: Path):
        """
        Internal prediction, can be called outside of mlflow
        """
        batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]

        adata = MLflowSCVI._filter_adata_by_hvg(adata, hvg_file)

        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )
        
        scvi.model.SCVI.prepare_query_anndata(
            adata, str(model_dir), return_reference_var_names=True
        )

        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(model_dir),
        )
        vae_q.is_trained = True
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)
        print(f"Latent representation: {qz_m[0:2, 0:2]}")
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

            del adata_filtered.varm
            
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