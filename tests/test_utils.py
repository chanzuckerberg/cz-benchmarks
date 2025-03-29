import os
import hydra
from omegaconf import OmegaConf
from czbenchmarks.utils import initialize_hydra, import_class_from_config


# Sample test class for import testing
class TestClass:
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2


def test_initialize_hydra():
    """Test hydra initialization with default and custom config paths."""
    # Test with default config path
    initialize_hydra()
    assert hydra.core.global_hydra.GlobalHydra.instance().is_initialized()

    # Clear hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Test with custom config path
    custom_path = os.path.join(os.path.dirname(__file__), "conf")
    initialize_hydra(custom_path)
    assert hydra.core.global_hydra.GlobalHydra.instance().is_initialized()

    # Clean up
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_import_class_from_config(tmp_path):
    """Test importing a class from a configuration file."""
    # Create a temporary config file
    config = {"_target_": "tests.test_utils.TestClass", "param1": "test", "param2": 42}

    config_path = tmp_path / "test_config.yaml"
    OmegaConf.save(config=config, f=config_path)

    # Import the class
    imported_class = import_class_from_config(str(config_path))

    # Verify it's the correct class
    assert imported_class == TestClass

    # Test that we can instantiate it with the config parameters
    instance = imported_class(param1="test", param2=42)
    assert instance.param1 == "test"
    assert instance.param2 == 42
