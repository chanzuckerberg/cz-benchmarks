import hydra
from omegaconf import OmegaConf
from czbenchmarks.datasets import utils as datasets_utils


def list_available_models() -> list[str]:
    """
    Lists all available models defined in the models.yaml configuration file.

    Returns:
        list: A sorted list of model names available in the configuration.
    """
    datasets_utils.initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="models"), resolve=True)

    # Extract model names
    model_names = list(cfg.get("models", {}).keys())

    # Sort alphabetically for easier reading
    model_names.sort()

    return model_names
