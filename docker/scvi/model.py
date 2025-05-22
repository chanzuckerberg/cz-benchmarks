import functools
import pathlib

import scvi
from omegaconf import OmegaConf
from utils import filter_adata_by_hvg

# from czbenchmarks.datasets import BaseDataset, DataType, Organism
# from czbenchmarks.models.implementations.base_model_implementation import (
#     BaseModelImplementation,
# )
# # from czbenchmarks.models.validators import BaseSingleCellValidator
# from czbenchmarks.utils import sync_s3_to_local
# from czbenchmarks.models.types import ModelType
from typing import Set


class SCVI():
    # def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
    #     return dataset.organism.name

    def _download_model_weights(self):
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)

        config = OmegaConf.load("/app/config.yaml")
        # TODO: hardcoded path to weights
        s3_path = config[dataset.organism.name]["model_dir"]
        bucket = s3_path.split("/")[2]
        path = "/".join(s3_path.split("/")[3:])
        sync_s3_to_local(bucket, path, str(model_dir))

    def run_model(self, input_file: pathlib.Path):
        adata = sc.read_h5ad(input_file)
        batch_keys = self.required_obs_keys


        # INFERENCE
        vae_q.is_trained = True

        qz_m, _ = vae_q.get_latent_representation(return_dist=True)


if __name__ == "__main__":
    SCVI().run()
