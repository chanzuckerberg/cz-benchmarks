# -----------------------------
# File: service.py
# -----------------------------
import bentoml
from bentoml.io import JSON
import yaml

from inference import SCVI

# Load artifact paths from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
artifacts_dict = config["artifacts"]

# Create an SCVI model instance
scvi_model = SCVI(artifacts_dict)

# Create the BentoML Service
svc = bentoml.Service("scvi_service")

# Define an API endpoint that accepts JSON input and returns JSON output
@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict):
    """
    Expects JSON input with keys:
      "file_path": path to an .h5ad file on disk
      "organism": (optional) string, e.g. "homo_sapiens" or "mus_musculus"
    """
    file_path = input_data.get("file_path")
    organism = input_data.get("organism", "homo_sapiens")
    
    # Run inference
    embedding = scvi_model.predict(file_path, {"organism": organism})
    
    # Return embedding as a JSON-serializable list
    return {"embedding": embedding.tolist()}