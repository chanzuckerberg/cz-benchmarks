import functools
import pathlib
import boto3
import pathlib
import scvi
import logging
from omegaconf import OmegaConf

from utils import filter_adata_by_hvg

# Import required classes from czibench
from czibench.models.sc import SingleCellModel
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism

logger = logging.getLogger(__name__)

class SCVI(SingleCellModel):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ['dataset_id', 'assay', 'suspension_type', 'donor_id']

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset) -> bool:
        # Check if all required batch keys are present in obs
        missing_keys = [key for key in cls.required_obs_keys if key not in dataset.adata.obs.columns]
        
        if missing_keys:
            logger.error(f"Missing required batch keys: {missing_keys}")
            return False
            
        return True
    
    def run_model(self):
        config = OmegaConf.load("config.yaml")

        adata = self.data.adata
        batch_keys = self.required_obs_keys
        adata = filter_adata_by_hvg(adata, f"hvg_names_{self.data.organism.name}.csv.gz")
        adata.obs["batch"] = functools.reduce(lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys])
        
        s3 = boto3.client('s3')
        model_dir = pathlib.Path("model")
        model_dir.mkdir(exist_ok=True)  # Create directory for model
        model_filename = model_dir / "model.pt"
        
        # Get S3 path based on organism
        s3_path = config[self.data.organism.name]["model_weights"]
        bucket = s3_path.split("/")[2]
        key = "/".join(s3_path.split("/")[3:])
        
        # Download the model file
        s3.download_file(bucket, key, str(model_filename))

        # Use the full path when loading the model
        scvi.model.SCVI.prepare_query_anndata(adata, str(model_dir), return_reference_var_names=True)

        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(model_dir),
        )
        vae_q.is_trained = True
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)

        self.data.output_embedding = qz_m    
        
if __name__ == "__main__":
    SCVI().run()