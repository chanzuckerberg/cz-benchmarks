import argparse
import pathlib

import scgpt as scg
from omegaconf import OmegaConf

from czbenchmarks.datasets import BaseDataset, DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.utils import sync_s3_to_local
from czbenchmarks.models.types import ModelType
from typing import Set


class ScGPTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGPT models.

    Validates datasets for use with Single-cell GPT models.
    Requires gene symbols and currently only supports human data.

    """

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_name"]
    model_type = ModelType.SCGPT

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData requirement
        """
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}


class ScGPT(ScGPTValidator, BaseModelImplementation):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_variant", type=str, default="human")
        args = parser.parse_args()
        return args

    def get_model_weights_subdir(self, _dataset: BaseDataset) -> str:
        args = self.parse_args()
        config = OmegaConf.load("config.yaml")
        selected_model = config.models[args.model_variant]
        model_variant = selected_model.model_name
        return model_variant

    def _download_model_weights(self, _dataset: BaseDataset):
        config = OmegaConf.load("config.yaml")
        args = self.parse_args()
        selected_model = config.models[args.model_variant]
        model_uri = selected_model.model_uri

        pathlib.Path(self.model_weights_dir).mkdir(exist_ok=True)

        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self, dataset: BaseDataset):
        adata = dataset.adata
        adata.var["gene_name"] = adata.var["feature_name"]
        ref_embed_adata = scg.tasks.embed_data(
            adata,
            model_dir=self.model_weights_dir,
            gene_col="gene_name",
            batch_size=32,
        )
        dataset.set_output(
            self.model_type, DataType.EMBEDDING, ref_embed_adata.obsm["X_scGPT"]
        )


if __name__ == "__main__":
    ScGPT().run()
