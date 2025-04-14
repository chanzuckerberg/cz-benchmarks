import glob
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from argparse import ArgumentParser, Namespace

from ...constants import (
    INPUT_DATA_PATH_DOCKER,
    MODEL_WEIGHTS_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    get_base_name,
    get_numbered_path,
)
from ...datasets import BaseDataset
from ..validators.base_model_validator import BaseModelValidator

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # This ensures the configuration is applied
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseModelImplementation(BaseModelValidator, ABC):
    """Abstract base class for model implementations.

    Handles common model operations like:
    - Loading datasets
    - Downloading model weights
    - Running inference
    - Saving results

    Attributes:
        datasets: List of datasets to process
        model_weights_dir: Directory containing model weights
    """

    datasets: List[BaseDataset]
    model_weights_dir: str

    def __init__(self, **user_kwargs):
        super().__init__()

        cli_args = self.parse_args()

        self.args = self._merge_arguments(
            cli_args=vars(cli_args), user_args=user_kwargs
        )

        self.args = Namespace(**self.args)

    @abstractmethod
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        """Get subdirectory for model variant weights.

        Args:
            dataset: Dataset being processed

        Returns:
            Subdirectory path or empty string if not needed
        """

    @abstractmethod
    def _download_model_weights(self, dataset: BaseDataset):
        """Download model weights if needed.

        Args:
            dataset: Dataset being processed
        """

    def download_model_weights(self, dataset: BaseDataset) -> None:
        """Download and verify model weights.

        Args:
            dataset: Dataset being processed
        """
        model_weights_subdir = self.get_model_weights_subdir(dataset)
        if model_weights_subdir != "":
            self.model_weights_dir = (
                f"{MODEL_WEIGHTS_PATH_DOCKER}/{model_weights_subdir}"
            )
        else:
            self.model_weights_dir = MODEL_WEIGHTS_PATH_DOCKER

        if not os.path.exists(self.model_weights_dir) or not any(
            os.listdir(self.model_weights_dir)
        ):
            logger.info("Downloading model weights...")
            self._download_model_weights(dataset)
            logger.info("Model weights downloaded successfully")
        else:
            logger.info("Model weights already downloaded...")

    @abstractmethod
    def run_model(self, dataset: BaseDataset) -> None:
        """Implement model-specific inference logic"""

    def parse_args(self) -> Namespace:
        """Centralized argument parsing using subclass-defined parser"""
        parser = self.create_parser()
        args = Namespace()
        if not parser:
            args = parser.parse_known_args()[0]
        return args

    def _merge_arguments(
        self, cli_args: Dict[str, Any], user_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge arguments with user input as priority"""
        return {**cli_args, **user_args}

    def create_parser(self) -> Optional[ArgumentParser]:
        """Subclasses implement to define their CLI arguments"""
        return None

    def run(self, datasets: Optional[BaseDataset | List[BaseDataset]] = None):
        """Run the full model pipeline.

        1. Load and validate all datasets
        2. Download model weights
        3. Run inference
        4. Save results

        Args:
            datasets: List of datasets to process

        Raises:
            FileNotFoundError: If no input datasets found
        """
        if datasets:
            serialize_datasets = False
            self.datasets = datasets if isinstance(datasets, list) else [datasets]
        else:
            serialize_datasets = True
            # Find all input datasets
            input_dir = pathlib.Path(INPUT_DATA_PATH_DOCKER).parent
            base_pattern = get_base_name(INPUT_DATA_PATH_DOCKER)
            input_files = sorted(glob.glob(os.path.join(input_dir, base_pattern)))

            if not input_files:
                raise FileNotFoundError("No input datasets found")

            # Ensure base file comes first if it exists
            if INPUT_DATA_PATH_DOCKER in input_files:
                input_files.remove(INPUT_DATA_PATH_DOCKER)
                input_files = [INPUT_DATA_PATH_DOCKER] + input_files

            # Load all datasets
            self.datasets = [
                self.dataset_type.deserialize(input_file) for input_file in input_files
            ]

        logger.info("Loading data...")
        for dataset in self.datasets:
            dataset.load_data()
        logger.info("Data loaded successfully")

        logger.info("Validating data...")
        for dataset in self.datasets:
            dataset.validate()
            self.validate_dataset(dataset)
        logger.info("Data validated successfully")

        for dataset in self.datasets:
            self.download_model_weights(dataset)

        logger.info("Running model...")
        for dataset in self.datasets:
            self.run_model(dataset)
        logger.info("Model ran successfully")

        if serialize_datasets:
            # Unload and serialize all datasets
            for i, dataset in enumerate(self.datasets):
                dataset.unload_data()
                output_path = get_numbered_path(OUTPUT_DATA_PATH_DOCKER, i)
                dataset.serialize(output_path)
