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

    available_organisms = [
        Organism.HUMAN,
        Organism.MOUSE,
        Organism.TROPICAL_CLAWED_FROG,
        Organism.AFRICAN_CLAWED_FROG,
        Organism.ZEBRAFISH,
        Organism.WILD_BOAR,
        Organism.RHESUS_MACAQUE,
        Organism.PLATYPUS,
        Organism.OPOSSUM,
        Organism.GORILLA,
        Organism.CHIMPANZEE,
        Organism.MARMOSET,
        Organism.CHICKEN,
        Organism.RABBIT,
        Organism.FRUIT_FLY,
        Organism.RAT,
        Organism.NAKED_MOLE_RAT,
        Organism.CAENORHABDITIS_ELEGANS,
        Organism.YEAST,
        Organism.MALARIA_PARASITE,
        Organism.SEA_LAMPREY,
        Organism.FRESHWATER_SPONGE,
        Organism.CORAL,
        Organism.SEA_URCHIN,
    ]
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
            "--model_variant",
            choices=["tf-sapiens", "tf-exemplar", "tf-metazoa"],
            default="tf-sapiens",
            help="TranscriptFormer model variant to use",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Batch size for inference. If None, uses model-specific defaults (sapiens: 32, exemplar: 8, metazoa: 2)",
        )

        return parser.parse_args()

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
        args = self.parse_args()
        model_variant = args.model_variant.replace("-", "_")
        batch_size = args.batch_size

        if batch_size is None:
            # These defaults have been emperically selected for fitting on a Tesla T4
            if model_variant == "tf_sapiens":
                batch_size = 32
            elif model_variant == "tf_exemplar":
                batch_size = 8
            elif model_variant == "tf_metazoa":
                batch_size = 2

        model_path = os.path.join(model_dir, model_variant)

        # Run inference using the inference.py script with Hydra configuration
        organism = dataset.organism.name

        cmd = [
            sys.executable,
            "transcriptformer/inference.py",
            "--config-name=inference_config.yaml",
            f"model.checkpoint_path={model_path}",
            f"model.inference_config.data_files.0={str(dataset.path)}",
            f"model.inference_config.batch_size={batch_size}",
            "model.inference_config.precision=16-mixed",
            f"model.inference_config.pretrained_embedding={str(model_dir)}/all_embeddings/{organism}_gene.h5",
        ]

        # Run the inference command
        subprocess.run(cmd, check=True)

        adata = anndata.read_h5ad("inference_results/embeddings.h5ad")
        embeddings = adata.obsm["embeddings"]

        # Set the output
        dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)


if __name__ == "__main__":
    TranscriptFormer().run()
