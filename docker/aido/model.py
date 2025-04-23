import functools
import pathlib
import argparse
import logging
from typing import Set

import anndata as ad
import numpy as np
import torch
from omegaconf import OmegaConf

from czbenchmarks.datasets import BaseDataset, DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.models.types import ModelType

from modelgenerator.tasks import Embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIDOValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = []  # e.g., could add keys such as "dataset_id", "assay", etc.
    required_var_keys = []
    model_type = ModelType.AIDO

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}


class AIDO(AIDOValidator, BaseModelImplementation):

    def parse_args(self):
        parser = argparse.ArgumentParser(description="AIDO Model Arguments")
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size for processing data (default: 2)"
        )
        parser.add_argument(
            "--model_variant",
            type=str,
            choices=["aido_cell_3m", "aido_cell_10m" "aido_cell_100m"],
            default="aido_cell_3m",
            help="Model variant to use (default: aido_cell_3m)"
        )
        args = parser.parse_args()
        return args

    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        pass

    def _download_model_weights(self, dataset: BaseDataset):
        pass

    def run_model(self, dataset: BaseDataset):
        try:
            args = self.parse_args() 
            batch_size = args.batch_size
            model_variant = args.model_variant
            adata: ad.AnnData = dataset.adata
            logger.info(f"AIDO: Starting model execution with batch size {batch_size}, model variant {model_variant} and Anndata shape {adata.shape}.")

            # Align the data to the AIDO.Cell gene set.
            import cell_utils 
            aligned_adata, attention_mask = cell_utils.align_adata(adata)

            device = "cuda" 

            # Load the AIDO model via the Embed class from ModelGenerator.
            model = Embed.from_config({
                "model.backbone": model_variant,
                "model.batch_size": batch_size
            }).eval()
            model = model.to(device).to(torch.float16)

            batch_np = aligned_adata[:batch_size].X.toarray()
            batch_tensor = torch.from_numpy(batch_np).to(torch.float16).to(device)
            batch_transformed = model.transform({'sequences': batch_tensor})
            # embs = model(batch_transformed)
            with torch.amp.autocast('cuda', torch.float16, True, True):
                embs = model(batch_transformed)

            # Replace any NaN values in the embeddings.
            if np.isnan(embs).any():
                print("AIDO: Detected NaN values in embeddings; replacing with zeros.")
                embs = np.nan_to_num(embs, nan=0.0)

            # Save the embeddings to the dataset’s output under the EMBEDDING data type.
            dataset.set_output(self.model_type, DataType.EMBEDDING, embs)
            logger.info("AIDO: Embeddings successfully saved to dataset output.")

        except Exception as e:
            logger.info("An error occurred during AIDO model execution:", e)
            raise


if __name__ == "__main__":
    # The run() method (inherited from BaseModelImplementation) will handle the
    # deserialization of the dataset (from INPUT_DATA_PATH_DOCKER) and execute run_model().
    AIDO().run()
