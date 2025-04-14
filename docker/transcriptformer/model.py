import pathlib
import subprocess
from typing import Set, Literal

from czbenchmarks.datasets import BaseDataset, DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.models.types import ModelType
from transcriptformer.inference import run_inference


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
        model_variant = self.parse_args()
        return f"{dataset.organism.name}_{model_variant}"

    def _download_model_weights(self, dataset: BaseDataset):
        model_variant = self.parse_args()
        model_dir = pathlib.Path(self.model_weights_dir)
        model_dir.mkdir(exist_ok=True)

        # Download model weights using TranscriptFormer's download script
        subprocess.run(
            [
                "python",
                "-m",
                "transcriptformer.download_artifacts",
                model_variant,
                f"--checkpoint-dir={str(model_dir)}",
            ],
            check=True,
        )

    def run_model(self, dataset: BaseDataset):
        model_path = str(self.model_weights_dir)

        # Prepare inference configuration
        inference_config = {
            "model.checkpoint_path": model_path,
            "model.inference_config.data_files.0": str(dataset.adata_path),
            "model.inference_config.batch_size": 32,
            "model.inference_config.precision": "16-mixed",
        }

        # Run inference
        embeddings = run_inference(inference_config)

        # Set the output
        dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)


if __name__ == "__main__":
    TranscriptFormer().run()
