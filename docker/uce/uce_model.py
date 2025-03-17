import argparse
import logging
import pathlib
import tempfile

import pandas as pd
from accelerate import Accelerator
from omegaconf import OmegaConf

from czbenchmarks.datasets import BaseDataset, DataType
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators.uce import UCEValidator
from czbenchmarks.utils import sync_s3_to_local

logger = logging.getLogger(__name__)


class UCE(UCEValidator, BaseModelImplementation):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_variant", type=str, default="4l")
        return parser.parse_args()

    def get_model_weights_subdir(self, _dataset: BaseDataset) -> str:
        return ""

    def _download_model_weights(self, _dataset: BaseDataset):
        config = OmegaConf.load("config.yaml")
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)

        model_uri = config.model_uri
        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self, dataset: BaseDataset):
        from evaluate import AnndataProcessor

        args = self.parse_args()
        model_variant = args.model_variant

        config = OmegaConf.load("config.yaml")
        assert model_variant in config.model_config, (
            f"Model {model_variant} not found in config.yaml. "
            f"Valid models are: {list(config.model_config.keys())}"
        )

        config.model_config[model_variant].protein_embeddings_dir = (
            f"{self.model_weights_dir}/protein_embeddings"
        )
        config.model_config[model_variant].model_loc = (
            f"{self.model_weights_dir}/"
            f"{config.model_config[model_variant].model_filename}"
        )
        config.model_config[model_variant].offset_pkl_path = (
            f"{self.model_weights_dir}/species_offsets.pkl"
        )
        config.model_config[model_variant].token_file = (
            f"{self.model_weights_dir}/all_tokens.torch"
        )
        config.model_config[model_variant].spec_chrom_csv_path = (
            f"{self.model_weights_dir}/species_chrom.csv"
        )

        # Create symbolic link for protein embeddings directory
        protein_embeddings_source = pathlib.Path(
            config.model_config[model_variant].protein_embeddings_dir
        )
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

        adata = dataset.adata
        adata.var_names = pd.Index(list(adata.var["feature_name"]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_adata_path = f"{tmp_dir}/temp_adata.h5ad"

            # Save adata to tempdir
            adata.write_h5ad(temp_adata_path)

            config.model_config[model_variant].adata_path = str(temp_adata_path)
            config.model_config[model_variant].dir = tmp_dir
            # where the embeddings are saved
            accelerator = Accelerator(project_dir=tmp_dir)
            config_dict = OmegaConf.to_container(
                config.model_config[model_variant], resolve=True
            )
            args = argparse.Namespace(**config_dict)
            processor = AnndataProcessor(args, accelerator)
            processor.preprocess_anndata()
            processor.generate_idxs()
            embedding_adata = processor.run_evaluation()
        dataset.set_output(
            self.model_type, DataType.EMBEDDING, embedding_adata.obsm["X_uce"]
        )


if __name__ == "__main__":
    UCE().run()
