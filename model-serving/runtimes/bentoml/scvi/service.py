# -----------------------------
# File: service.py
# -----------------------------

import bentoml
import yaml
from inference import SCVI


@bentoml.service(
    name="scvi_service",
    resources={"cpu": "2"},
    traffic={"timeout": 300},  # Increased timeout for potentially large datasets
)
class SCVIService:
    def __init__(self) -> None:
        # Load artifact paths from config.yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        artifacts_dict = config["artifacts"]

        # Create an SCVI model instance
        self.model = SCVI(artifacts_dict)

    @bentoml.task
    def predict(self, file_path: str, organism: str = "homo_sapiens") -> dict:
        """
        Expects JSON input with keys:
          "file_path": path to an .h5ad file on disk
          "organism": (optional) string, e.g. "homo_sapiens" or "mus_musculus"
        """

        # Run inference
        embedding = self.model.predict(file_path, {"organism": organism})

        # Return embedding as a JSON-serializable list
        return {"embedding": embedding.tolist()}
