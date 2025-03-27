from typing import List
import hydra
from omegaconf import OmegaConf

from ..datasets.utils import initialize_hydra


def list_available_models() -> List[str]:
    """
    Lists all available models defined in the models.yaml configuration file.

    Returns:
        list: A sorted list of model names available in the configuration.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="models"), resolve=True)

    # Extract dataset names
    model_names = list(cfg.get("models", {}).keys())

    # Sort alphabetically for easier reading
    model_names.sort()

    return model_names
