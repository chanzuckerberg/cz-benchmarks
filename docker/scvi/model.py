import functools
import pathlib

import scvi
from omegaconf import OmegaConf
from utils import filter_adata_by_hvg

from czbenchmarks.datasets import BaseDataset, DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.utils import sync_s3_to_local
from czbenchmarks.models.types import ModelType
from typing import Set


class SCVIValidator(BaseSingleCellValidator):
    """Validation requirements for scVI models.

    Validates datasets for use with Single-cell Variational Inference models.
    Requires detailed metadata about the dataset, assay, and donor information.
    Supports both human and mouse data.

    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []
    model_type = ModelType.SCVI

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData and metadata requirements
        """
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}


class SCVI(SCVIValidator, BaseModelImplementation):
    def parse_args(self):
        pass

    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        return dataset.organism.name

    def _download_model_weights(self, dataset: BaseDataset):
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)

        config = OmegaConf.load("config.yaml")
        s3_path = config[dataset.organism.name]["model_dir"]
        bucket = s3_path.split("/")[2]
        path = "/".join(s3_path.split("/")[3:])
        sync_s3_to_local(bucket, path, str(model_dir))

    def run_model(self, dataset: BaseDataset):
        adata = dataset.adata
        batch_keys = self.required_obs_keys
        adata = filter_adata_by_hvg(
            adata, f"{self.model_weights_dir}/hvg_names_{dataset.organism.name}.csv.gz"
        )
        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )

        scvi.model.SCVI.prepare_query_anndata(
            adata, str(self.model_weights_dir), return_reference_var_names=True
        )
        vae_q = scvi.model.SCVI.load_query_data(adata, str(self.model_weights_dir))
        vae_q.is_trained = True
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)

        dataset.set_output(self.model_type, DataType.EMBEDDING, qz_m)


if __name__ == "__main__":
    SCVI().run()
