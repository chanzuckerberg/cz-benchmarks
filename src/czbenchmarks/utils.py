import logging
import os
import yaml
from typing import Any, Dict, Optional

import hydra
from omegaconf import OmegaConf, open_dict

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DEFAULT_HYDRA_CONFIG_PATH = "./conf"


def initialize_hydra(config_path=DEFAULT_HYDRA_CONFIG_PATH):
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


def load_custom_config(
    item_name: str,
    config_name: str,
    custom_config_path: Optional[str] = None,
    class_update_kwargs: Optional[Dict[str, Any]] = None,
):
    """Customize czbenchmarks parameters for class instantiation

    Args:
        config_name: Name of the czbenchmarks config to load, e.g. "datasets" for "datasets.yaml".
        item_name: Item from the czbenchmarks config to load, e.g. "replogle_k562_essential_perturbpredict" for "datasets.yaml".
            If the item is not in the config, a new item will be created.
        custom_config_path: Optional path to a YAML file containing a custom configuration that can be used to update the existing
            default configuration.
        class_update_kwargs: Optional dictionary of parameters to update those used for class instantiation

    Returns:
        Configuration
    """

    strict_checking_is_disabled = False
    initialize_hydra()
    cfg = hydra.compose(config_name=config_name)

    # If custom config provided, load and merge it
    if custom_config_path:
        # Expand user path (handles ~)
        custom_config_path = os.path.expanduser(custom_config_path)
        custom_config_path = os.path.abspath(custom_config_path)

        if not os.path.exists(custom_config_path):
            raise FileNotFoundError(
                f"Custom config file not found: {custom_config_path}"
            )
        else:
            logger.info(
                f"Updating czbenchmarks config with data from custom yaml config: {custom_config_path}"
            )

        with open(custom_config_path) as f:
            custom_cfg = OmegaConf.create(yaml.safe_load(f))

        # Disable strict checking and merge configs
        # TODO check if custom_cfg introduces new keys and only disable strict checking if needed
        if strict_checking_is_disabled:
            log_msg = "Strict checking already disabled "
        else:
            log_msg = "Disabled strict checking "
            strict_checking_is_disabled = True
            OmegaConf.set_struct(cfg, False)
        logger.info(
            log_msg
            + f'to allow updating "{config_name}" with custom yaml config "{custom_config_path}" in config "{config_name}"'
        )

        cfg = OmegaConf.merge(cfg, custom_cfg)

    # Adding new dictionary keys to the config if provided
    if class_update_kwargs:
        # Handle case where item_name is not in the config
        if item_name not in cfg[config_name]:
            if strict_checking_is_disabled:
                log_msg = "Strict checking already disabled "
            else:
                log_msg = "Disabled strict checking "
                strict_checking_is_disabled = True
                OmegaConf.set_struct(cfg, False)
            logger.info(
                log_msg + f'to allow creating "{item_name}" under "{config_name}"'
            )

            with open_dict(cfg):
                cfg[config_name][item_name] = {}

            updated_keys, new_keys = None, class_update_kwargs.keys()

        else:
            updated_keys = (
                class_update_kwargs.keys() & cfg[config_name][item_name].keys()
            )
            new_keys = class_update_kwargs.keys() - cfg[config_name][item_name].keys()

        # Log all changes made to the config for user visibility
        if updated_keys:
            update_keys_str = ", ".join(
                [
                    f"{key} from {cfg[config_name][item_name][key]} to {class_update_kwargs[key]}"
                    for key in updated_keys
                ]
            )
            logger.info(
                f'Updating the following config items under "{item_name}": {update_keys_str}'
            )

        if new_keys:
            new_keys_str = ", ".join(
                [f"{key}={class_update_kwargs[key]}" for key in new_keys]
            )
            logger.info(
                f'Adding the following new items to "{item_name}": {new_keys_str}'
            )
            if strict_checking_is_disabled:
                log_msg = "Strict checking already disabled "
            else:
                log_msg = "Disabled strict checking "
                strict_checking_is_disabled = True
                OmegaConf.set_struct(cfg, False)
            logger.info(
                log_msg + f'to allow adding these keys to the config "{item_name}"'
            )

        cfg[config_name][item_name] = OmegaConf.merge(
            cfg[config_name][item_name], class_update_kwargs
        )

    custom_cfg = cfg[config_name][item_name]
    return custom_cfg
