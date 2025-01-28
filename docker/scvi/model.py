import functools
import pathlib
import boto3
import pathlib
import scvi
import logging
from omegaconf import OmegaConf

from utils import filter_adata_by_hvg

from czibench.models.sc import ScviValidator

logger = logging.getLogger(__name__)


class SCVI(ScviValidator):
    
    def get_model_weights_subdir(self) -> str:
        return self.data.organism.name
    
    def _download_model_weights(self):
        s3 = boto3.client("s3")
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)  # Create directory for model
        model_filename = model_dir / "model.pt"

        # Get S3 path based on organism
        config = OmegaConf.load("config.yaml")
        s3_path = config[self.data.organism.name]["model_weights"]
        bucket = s3_path.split("/")[2]
        key = "/".join(s3_path.split("/")[3:])

        s3.download_file(bucket, key, str(model_filename))
                    
    def run_model(self):

        adata = self.data.adata
        batch_keys = self.required_obs_keys
        adata = filter_adata_by_hvg(
            adata, f"hvg_names_{self.data.organism.name}.csv.gz"
        )
        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )

        # Use the full path when loading the model
        scvi.model.SCVI.prepare_query_anndata(
            adata, str(self.model_weights_dir), return_reference_var_names=True
        )

        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(self.model_weights_dir),
        )

        vae_q.is_trained = True
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)

        self.data.output_embedding = qz_m


if __name__ == "__main__":
    SCVI().run()
