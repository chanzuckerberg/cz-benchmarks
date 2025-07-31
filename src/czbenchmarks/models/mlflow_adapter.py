import mlflow
import os
import logging
from typing import Dict, Any
from . import ModelAdapter

logger = logging.getLogger(__name__)

class MLflowAdapter(ModelAdapter):
    """
    Adapter to run model inference using MLflow artifacts.
    """

    def __init__(self, model_uri: str, config: Dict[str, Any] = None):
        """
        Initialize the MLflowAdapter with the URI of the MLflow model.
        """
        if config is None:
            config = {}
        config["model_uri"] = model_uri
        super().__init__(config)

    def _validate_config(self):
        if "model_uri" not in self.config:
            raise ValueError("Config must include 'model_uri'.")

    def _setup(self):
        """
        Setup any required dependencies for MLflow.
        """
        pass

    def run(self, input_data: Any) -> Any:
        """
        Run inference using the MLflow model.
        """
        model_uri = self.config.get("model_uri")

        if not model_uri:
            raise ValueError("'model_uri' must be specified in config.")

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            result = model.predict(input_data)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.exception("Error while running MLflow model adapter")
            raise RuntimeError(f"Error while running MLflow model adapter: {exc}") from exc

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Run inference using MLflowAdapter.")
    parser.add_argument("--model-uri", required=True, help="URI of the MLflow model.")
    parser.add_argument("--input", required=True, help="Path to the input data (CSV format).")

    args = parser.parse_args()

    input_data = pd.read_csv(args.input)

    adapter = MLflowAdapter(args.model_uri)
    result = adapter.run(input_data)
    print(result)