import argparse
import logging
import os
import pathlib
import tempfile

import pandas as pd
from accelerate import Accelerator

import argparse
from omegaconf import OmegaConf
import tempfile
import os
from czibench.utils import sync_s3_to_local
from czibench.datasets.types import Organism
from czibench.datasets.sc import SingleCellDataset
from czibench.models.sc import BaseSingleCell

logger = logging.getLogger(__name__)


class UCE(BaseSingleCell):

    available_organisms = [Organism.HUMAN, Organism.MOUSE] # TODO: add other UCE organisms
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        missing_keys = [
            key for key in cls.required_var_keys if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")

        return True
    
    def get_model_weights_subdir(self) -> str:
        return ""
    
    def _download_model_weights(self):
        config = OmegaConf.load("config.yaml")
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True) 
        
        model_uri = config.model_uri
        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])
        
        sync_s3_to_local(bucket, key, self.model_weights_dir)
        
    def run_model(self):
        from evaluate import AnndataProcessor

        config = OmegaConf.load("config.yaml")

        config.model_config.protein_embeddings_dir = f"{self.model_weights_dir}/protein_embeddings"
        config.model_config.model_loc = f"{self.model_weights_dir}/{config.model_config.model_filename}"
        config.model_config.offset_pkl_path = f"{self.model_weights_dir}/species_offsets.pkl"
        config.model_config.token_file = f"{self.model_weights_dir}/all_tokens.torch"
        config.model_config.spec_chrom_csv_path = f"{self.model_weights_dir}/species_chrom.csv"

        # Create symbolic link for protein embeddings directory
        protein_embeddings_source = pathlib.Path(config.model_config.protein_embeddings_dir)

        protein_embeddings_target = pathlib.Path("model_files/protein_embeddings")
        protein_embeddings_target.parent.mkdir(parents=True, exist_ok=True)
        if protein_embeddings_target.exists():
            protein_embeddings_target.unlink()
        protein_embeddings_target.symlink_to(protein_embeddings_source)

        print(f"Contents of {protein_embeddings_target}:\n")
        if protein_embeddings_target.exists():
            for path in protein_embeddings_target.rglob("*"):
                print(f"{path.relative_to(protein_embeddings_target)}\n")
        else:
            print("Directory does not exist\n")

        adata = self.data.adata
        adata.var_names = pd.Index(list(adata.var["feature_name"]))
        tmp_dir = pathlib.Path(tempfile.gettempdir()) / "temp_adata"
        os.makedirs(tmp_dir, exist_ok=True)
        temp_adata_path = tmp_dir / "temp_adata.h5ad"

        # Save adata to tempdir
        adata.write_h5ad(temp_adata_path)

        # set features to be gene symbols which is required
        # by required by evaluate.AnndataProcessor
        adata.var_names = adata.var["feature_name"].values
        config.model_config.adata_path = str(temp_adata_path)

        # where the embeddings are saved
        accelerator = Accelerator(project_dir=".")
        config_dict = OmegaConf.to_container(config.model_config, resolve=True)
        args = argparse.Namespace(**config_dict)
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        embedding_adata = processor.run_evaluation()
        self.data.output_embedding = embedding_adata.X.toarray()


if __name__ == "__main__":
    UCE().run()
