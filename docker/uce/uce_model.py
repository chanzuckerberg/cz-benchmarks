import pathlib
import boto3
import pathlib
import logging
import anndata as ad
from accelerate import Accelerator
from czibench.models.sc import UCEValidator
import argparse
from omegaconf import OmegaConf
import sys
import tempfile
import os
logger = logging.getLogger(__name__)


class UCE(UCEValidator):

    def run_model(self):
        from evaluate import AnndataProcessor

        config = OmegaConf.load("config.yaml")

        adata = self.data.adata
        tmp_dir = pathlib.Path(tempfile.gettempdir()) / "temp_adata"
        os.makedirs(tmp_dir, exist_ok=True)
        temp_adata_path = tmp_dir / f"temp_adata.h5ad"

        # Save adata to tempdir
        adata.write_h5ad(temp_adata_path)

        # set features to be gene symbols which is required by required by evaluate.AnndataProcessor
        adata.var_names = adata.var["feature_name"].values
        config.model_config.adata_path = str(temp_adata_path)

        accelerator = Accelerator(project_dir=".")  # where the embeddings are saved
        config_dict = OmegaConf.to_container(config.model_config, resolve=True)
        args = argparse.Namespace(**config_dict)
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        embedding_adata = processor.run_evaluation()
        self.data.output_embedding = embedding_adata.X.toarray()

if __name__ == "__main__":
    UCE().run()
