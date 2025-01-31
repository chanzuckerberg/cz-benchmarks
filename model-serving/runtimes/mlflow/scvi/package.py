import mlflow
import mlflow.pyfunc
from mlflow.types import Schema, ColSpec, ParamSchema, ParamSpec, DataType

from mlflow_scvi import MLflowSCVI
from pathlib import Path
import shutil


if __name__ == "__main__":
    # Save the model
    mlflow_model_path = "runtime"

    if Path(mlflow_model_path).exists():
        shutil.rmtree(mlflow_model_path)
        
    mlflow.pyfunc.save_model(
        path=mlflow_model_path, 
        python_model=MLflowSCVI(),

        # Note: even though the model object is pickled and stored, the source code is still needed to avoid errors at runtime
        code_paths=["mlflow_scvi.py"],
        # infer_code_paths=True,
        signature=mlflow.models.ModelSignature(
            inputs=Schema([ColSpec("binary")]),
            outputs=Schema([ColSpec("binary")]),
            params=ParamSchema([ParamSpec("organism", DataType.string, default="homo_sapiens")])),
        artifacts={
            # TODO: download artifact files from s3, not local filesystem
            # NOTE: It's non-standard for a model to have multiple sets of weights
            "model_weights_homo_sapiens": "artifacts/homo_sapiens/model.pt",
            "model_weights_mus_musculus": "artifacts/mus_musculus/model.pt",
            "hvg_names_homo_sapiens": "artifacts/homo_sapiens/hvg_names.csv.gz",
            "hvg_names_mus_musculus": "artifacts/mus_musculus/hvg_names.csv.gz"
            },
        extra_pip_requirements="requirements-model.txt"
    )
    
    print(f"Model saved to {mlflow_model_path}")
    
