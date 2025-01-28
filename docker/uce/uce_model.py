import argparse
import logging
import os
import pathlib
import tempfile

import pandas as pd
from accelerate import Accelerator
from omegaconf import OmegaConf

from czibench.models.sc import UCEValidator

logger = logging.getLogger(__name__)


class UCE(UCEValidator):
    def run_model(self):
        from evaluate import AnndataProcessor

        config = OmegaConf.load("config.yaml")

        # Create symbolic link for protein embeddings directory
        protein_embeddings_source = pathlib.Path(config.paths.embedding_dir)
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

        # set features to be gene symbols which is required by required by
        # evaluate.AnndataProcessor
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
