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
import os
logger = logging.getLogger(__name__)


class UCE(UCEValidator):

    def run_model(self):
        from evaluate import AnndataProcessor

        config = OmegaConf.load("config.yaml")

        adata = self.data.adata
        adata.var_names = adata.var["feature_name"].values
        model_loc = pathlib.Path(config.model_config.model_loc)

        config.model_config.adata_path = adata.path
        config.model_config.dir = str(config.paths.data_dir) + f"/{eval_set_name}/"
        os.makedirs(config.model_config.dir, exist_ok=True)

        accelerator = Accelerator(project_dir=".")  # where the embeddings are saved
        config_dict = OmegaConf.to_container(config.model_config, resolve=True)
        args = argparse.Namespace(**config_dict)
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        embedding_adata = processor.run_evaluation()
        output_adata = ad.AnnData(
            X=None,
            obsm={"emb": embedding_adata.X.toarray()},
            obs=adata.obs,
            var=adata.var,
        )
        logger.info(f"Output embedding shape: {output_adata.obsm['emb'].shape}")

        self.data.output_embedding = output_adata.obsm["emb"]



if __name__ == "__main__":
    UCE().run()
