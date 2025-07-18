import logging

import hydra
from omegaconf import OmegaConf


logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def initialize_hydra(config_path="./conf"):
    """
    Initialize Hydra configuration system.

    This function sets up Hydra's configuration system using the specified
    configuration directory. If Hydra is already initialized, it clears the
    existing instance before reinitializing.

    Args:
        config_path (str): Path to the configuration directory.

    Returns:
        None
    """
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(
        config_path=config_path,
        version_base=None,
    )


def import_class_from_config(config_path: str):
    """
    Import a class based on the `_target_` field in a configuration file.

    This function reads a configuration file, extracts the `_target_` field,
    and dynamically imports the specified class.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        class_obj: The imported class object.

    Raises:
        AttributeError: If the specified class does not exist in the module.
        ImportError: If the module cannot be imported.
    """
    # Load the configuration
    logger.info(f"Loading model configuration from {config_path}")
    cfg = OmegaConf.load(config_path)

    # Get the target class path
    target_path = cfg._target_

    # Import the class using the target path
    module_path, class_name = target_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    class_obj = getattr(module, class_name)

    logger.info(f"Imported class: {class_obj.__name__}")

    return class_obj
