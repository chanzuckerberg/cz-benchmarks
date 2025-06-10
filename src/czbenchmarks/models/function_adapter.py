import importlib.util
import os
import logging
from typing import Dict, Any
from . import ModelAdapter

logger = logging.getLogger(__name__)

class FunctionAdapter(ModelAdapter):
    """
    Adapter to call a Python function for running model inference. This adapter allows
    execution of a specified function from a Python file, enabling flexible integration
    with existing Python codebases without needing to modify them.
    """

    def __init__(self, file_path: str, function_name: str = "run", config: Dict[str, Any] = None):
        """
        Initialize the FunctionAdapter with the path to the Python file and the function name.
        If function_name is not provided, defaults to '__main__'.
        """
        if config is None:
            config = {}
        config["file_path"] = file_path
        config["function_name"] = function_name
        super().__init__(config)

    def _validate_config(self):
        if "file_path" not in self.config: # or "function_name" not in self.config:
            raise ValueError("Config must include 'file_path' and 'function_name'.")

    def _setup(self):
        # TODO: implement setup of virtual environment or dependencies if needed
        pass

    def run(self, input_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, str]:
        file_path = self.config.get("file_path")
        function_name = self.config.get("function_name")

        if not file_path or not function_name:
            raise ValueError("Both 'file_path' and 'function_name' must be specified in config.")

        try:
            spec = importlib.util.spec_from_file_location("external_module", file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec from {file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func = getattr(module, function_name, None)
            if func is None or not callable(func):
                if function_name == "__main__":
                    if hasattr(module, "__file__"):
                        exec(open(module.__file__).read(), {"__name__": "__main__", "input_path": input_path, "output_path": output_path, "params": params})
                        return {"status": "success", "result": "Executed __main__ block"}
                    else:
                        raise AttributeError(f"No __main__ block found in '{file_path}'.")
                else:
                    raise AttributeError(f"Function '{function_name}' not found in '{file_path}'.")

            result = func(input_path, output_path, params)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.exception("Error while running Python function adapter")
            raise RuntimeError(f"Error while running Python function adapter: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a Python function from a file using FunctionAdapter.")
    parser.add_argument("--file", required=True, help="Path to the Python file containing the function.")
    parser.add_argument("--function", default="__main__", help="Name of the function to call (default: __main__).")
    parser.add_argument("--input", required=True, help="Input path argument for the function.")
    parser.add_argument("--output", required=True, help="Output path argument for the function.")
    parser.add_argument("--params", default="{}", help="JSON string of additional parameters.")

    args = parser.parse_args()

    import json
    params = json.loads(args.params)

    adapter = FunctionAdapter(args.file, args.function)
    result = adapter.run(args.input, args.output, params)
    print(result)