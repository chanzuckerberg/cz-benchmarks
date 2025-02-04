import os
import mlflow
import mlflow.models
import mlflow.pyfunc
from mlflow.types import Schema, ColSpec, ParamSchema, ParamSpec, DataType

from mlflow_scvi import MLflowSCVI
from pathlib import Path
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save and log MLflow model")
    parser.add_argument("--target", choices=["local", "databricks"], default="local", required=True, help="Target to save the model: 'local' or 'databricks'")
    parser.add_argument("--experiment-name", type=str, required=False, help="Name of the MLflow experiment")
    args = parser.parse_args()
    if args.target == "databricks" and not args.experiment_name:
        parser.error("--experiment-name is required when target is 'databricks'")
    
    artifacts={
        "model_weights_homo_sapiens": "artifacts/homo_sapiens/model.pt",
        # "model_weights_mus_musculus": "artifacts/mus_musculus/model.pt",
        "hvg_names_homo_sapiens": "artifacts/homo_sapiens/hvg_names.csv.gz",
        # "hvg_names_mus_musculus": "artifacts/mus_musculus/hvg_names.csv.gz"
    }
    signature=mlflow.models.ModelSignature(
                    inputs=Schema([ColSpec("string")]),
                    outputs=Schema([ColSpec("string")]),
                    params=ParamSchema([ParamSpec("organism", DataType.string, default="homo_sapiens")])
                )
    code_paths = ["mlflow_scvi.py"]
    

    if args.target == "local":
        # Save the model locally
        mlflow_model_path = "runtime"
        if Path(mlflow_model_path).exists():
            shutil.rmtree(mlflow_model_path)
            
        mlflow.pyfunc.save_model(
            path=mlflow_model_path, 
            python_model=MLflowSCVI(),
            code_paths=code_paths,
            signature=signature,
            artifacts=artifacts,
            extra_pip_requirements="requirements-model.txt"
        )
        print(f"Model saved locally to {mlflow_model_path}")

    elif args.target == "databricks":
        # Create an MLflow experiment in Databricks. 
        # See https://docs.databricks.com/en/mlflow/models.html#log-load-and-register-mlflow-models
        # Requires Databricks Tracking Server configuration: https://docs.databricks.com/en/mlflow/access-hosted-tracking-server.html

        # Note: Do NOT use /Workspace prefix in experiment name; it is implied. While the experiment will be recorded, later get_experiment_by_name() calls will fail to find the experiment. (https://github.com/mlflow/mlflow/issues/11077)
        if experiment := mlflow.get_experiment_by_name(args.experiment_name):
            experiment_id = experiment.experiment_id
            print(f"Using experiment id={experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(name=args.experiment_name)
            print(f"Created experiment id={experiment_id}")
        
        # Log the model to Databricks tracking server
        with mlflow.start_run(experiment_id=experiment_id):
            model_info = mlflow.pyfunc.log_model(
                artifact_path="artifacts",
                python_model=MLflowSCVI(),
                code_paths=code_paths,
                signature=signature,
                artifacts=artifacts,
                extra_pip_requirements="requirements-model.txt"
            )
        print(f"Model logged to Databricks at {args.experiment_name}")
        
        # Register the model
        model_version = mlflow.register_model(model_info.model_uri, "scvi")
        
        # TODO: Serve the model (this can be done via the Databricks web console as well)
        
        

    
