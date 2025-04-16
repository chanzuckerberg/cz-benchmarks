import pathlib
import os
import subprocess
import sys
from typing import Set

import anndata
from czbenchmarks.datasets import BaseDataset, DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.models.types import ModelType


class TranscriptFormerValidator(BaseSingleCellValidator):
    """Validation requirements for TranscriptFormer models.

    Validates datasets for use with TranscriptFormer models.
    Requires gene IDs in Ensembl format and supports both human and mouse data.
    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = []
    required_var_keys = ["ensembl_id"]
    model_type = ModelType.TRANSCRIPTFORMER

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData and metadata requirements
        """
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}


class TranscriptFormer(TranscriptFormerValidator, BaseModelImplementation):
    def __init__(self):
        super().__init__()

        # TranscriptFormer operates directly on data files stored on disk rather than loading data into memory
        self.load_data = False

    def parse_args(self):
        """Parse command line arguments to select model variant.

        Available variants:
        - tf-sapiens: Default model for human data
        - tf-exemplar: Model trained on exemplar species
        - tf-metazoa: Model trained on metazoan species
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model-variant",
            choices=["tf-sapiens", "tf-exemplar", "tf-metazoa"],
            default="tf-sapiens",
            help="TranscriptFormer model variant to use",
        )
        args = parser.parse_args()
        return args.model_variant

    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        return ""

    def _download_model_weights(self, dataset: BaseDataset):
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)

        # Download model weights using TranscriptFormer's download script
        subprocess.run(
            [
                sys.executable,
                "transcriptformer/download_artifacts.py",
                "all",
                f"--checkpoint-dir={str(model_dir)}",
            ],
            check=True,
        )

    def run_model(self, dataset: BaseDataset):
        model_dir = str(self.model_weights_dir)

        # Get model variant
        model_variant = self.parse_args().replace("-", "_")

        model_path = os.path.join(model_dir, model_variant)

        # Run inference using the inference.py script with Hydra configuration
        cmd = [
            sys.executable,
            "transcriptformer/inference.py",
            "--config-name=inference_config.yaml",
            f"model.checkpoint_path={model_path}",
            f"model.inference_config.data_files.0={str(dataset.path)}",
            "model.inference_config.batch_size=32",
            "model.inference_config.precision=16-mixed",
        ]

        # Run the inference command
        subprocess.run(cmd, check=True)

        adata = anndata.read_h5ad("transcriptformer/inference_results/embeddings.h5ad")
        embeddings = adata.obsm["embeddings"]

        # Set the output
        dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)


if __name__ == "__main__":
    TranscriptFormer().run()
