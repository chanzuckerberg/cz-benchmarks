import abc
import logging
from typing import Dict, Any, Optional, List
import os
import time
import json
import yaml

logger = logging.getLogger(__name__)


class ModelAdapter(abc.ABC):
    """
    Abstract base class for all inference adapters.

    ModelAdapter provides a standardized interface and common functionality for loading configurations,
    validating them, and setting up resources required for inference tasks. All concrete adapters must
    implement the `run()` method to define their inference logic.

    Key Methods:
        - _load_config():
            Loads configuration settings from a provided dictionary or from a JSON/YAML configuration file.
            Merges loaded settings into the adapter's configuration dictionary.
        - _validate_config():
            Validates the configuration dictionary. This method is intended to be overridden by subclasses
            to enforce required configuration keys or value constraints.
        - _setup():
            Optional setup step for initializing resources or performing any pre-inference preparation.
            Subclasses can override this method as needed.
        - run():
            Abstract method that must be implemented by subclasses. Executes the inference process using the
            provided input and parameters, and returns a dictionary mapping artifact keys to file paths.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize the ModelAdapter.

        Args:
            config: Optional dictionary of configuration parameters.
            config_file: Optional path to a JSON or YAML configuration file.
        """
        self.config = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_config(config=config, config_file=config_file)
        self._setup()

    def _load_config(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None) -> None:
        """
        Load and update the configuration dictionary.

        Can accept a config dict directly, or load from a JSON/YAML file.
        If both are provided, values from the config dict override those from the file.

        Args:
            config: Optional dictionary of configuration parameters.
            config_file: Optional path to a JSON or YAML configuration file.
        """
        if config_file:
            ext = os.path.splitext(config_file)[1].lower()
            with open(config_file, "r") as f:
                if ext in [".yaml", ".yml"]:
                    try:
                        import yaml
                    except ImportError:
                        raise ImportError("PyYAML is required to load YAML config files.")
                    file_config = yaml.safe_load(f)
                elif ext == ".json":
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file extension: {ext}")
            self.config.update(file_config)
        if config:
            self.config.update(config)
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration dictionary.
        Subclasses can override this to enforce required config keys.
        """
        pass

    def _setup(self):
        """
        Optional setup step.
        Subclasses can override to initialize resources.
        """
        pass

    @abc.abstractmethod
    def run(
        self, 
        input_path: str, 
        output_path: str, 
        params: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Run inference. Return a dict mapping artifact keys → file paths (e.g. {"predictions": "/tmp/out.json"}).
        Subclasses must implement this method.
        """
        pass


    def _cleanup(self):
        """
        Optional cleanup step to remove temporary files or reset state.
        Subclasses can override this method.
        """
        pass
