from pathlib import Path
import shutil
import mlflow
import mlflow.models
import mlflow.pyfunc
from mlflow.types import Schema, ColSpec, ParamSchema, ParamSpec, DataType
import numpy as np

from mlflow_scvi import MLflowSCVI
import argparse

def save_mlflow_model_locally():
    if not Path(mlflow_model_path).exists() or args.force_package_build:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save and log MLflow model")
    parser.add_argument("--target", choices=["local", "databricks", "sagemaker", "sagemaker-local"], default="local", required=True, help="Target to save the model: 'local' or 'databricks'")
    parser.add_argument("--experiment-name", type=str, required=False, help="Name of the MLflow experiment")
    parser.add_argument("--force-package-build", action="store_true", help="Force recreate the MLflow model even if it exists")
    parser.add_argument("--use-existing-package", action="store_false", dest="force", default=True, help="Do not force recreate the MLflow model if it exists")
    args = parser.parse_args()
    if args.target == "databricks" and not args.experiment_name:
        parser.error("--experiment-name is required when target is 'databricks'")
    
    artifacts={
        "model_weights_homo_sapiens": "artifacts/homo_sapiens_model.pt",
        "model_weights_mus_musculus": "artifacts/mus_musculus_model.pt",
        "hvg_names_homo_sapiens": "artifacts/homo_sapiens_hvg_names.csv.gz",
        "hvg_names_mus_musculus": "artifacts/mus_musculus_hvg_names.csv.gz"
    }
    signature=mlflow.models.ModelSignature(
                    inputs=Schema([ColSpec("string")]),
                    outputs=Schema([ColSpec("string")]),
                    params=ParamSchema([ParamSpec("organism", DataType.string, default="homo_sapiens")])
                )
    code_paths = ["mlflow_scvi.py"]
    mlflow_model_path = "runtime"


    if args.target == "local":
        save_mlflow_model_locally()

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
        
        # Check if the model already exists for this experiment
        existing_models = mlflow.search_registered_models(filter_string="name='scvi'")

        if args.force or not existing_models:
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

    elif args.target in ["sagemaker", "sagemaker-local"]:
        from sagemaker.serve import ModelBuilder
        from sagemaker.serve.mode.function_pointers import Mode
        from sagemaker.serve import SchemaBuilder
        
        if args.target.endswith("local"):
            sm_deploy_mode = Mode.LOCAL_CONTAINER
            sm_instance_type = "local"            
        else:
            sm_deploy_mode = Mode.SAGEMAKER_ENDPOINT
            sm_instance_type = "ml.g4dn.xlarge"

        save_mlflow_model_locally()
                                    
        model_schema = SchemaBuilder(
            sample_input="s3://path/input.h5ad",
            sample_output=np.array([[0.0020090353209525347, 0.000620177364908158]])
        )
        
        # Note: Requires AWS_PROFILE or AWS_DEFAULT_REGION environment variable is set
        # Note: To ensure use of GPU Docker base image, must specify instance_type
        # TODO: How do we specify async endpoint?
        model_builder = ModelBuilder(
            name="scvi-mlflow",
            mode=sm_deploy_mode,
            # mode=Mode.SAGEMAKER_ENDPOINT,
            instance_type=sm_instance_type,
            # FIXME: A schema is required to avoid failing with "cannot serialize" error; 
            # however, the model is actually being invoked on deploy with the schema's sample inputs--why!?
            schema_builder=model_schema,
            # TODO: This is assumed to have been created already by having run model-serving/runtimes/sagemaker/scvi/remote/deploy.py (remote async sagemaker endpoint)
            role_arn="arn:aws:iam::058264139299:role/OmarSageMakerRole",
            model_metadata={
                # both model path and tracking server ARN are required if you use an mlflow run ID or mlflow model registry path as input
                "MLFLOW_MODEL_PATH": mlflow_model_path,
                # "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:region:account-id:mlflow-tracking-server/tracking-server-name"
            }
        )
        # TODO: The `build` step takes many minutes!?
        model = model_builder.build()

        # This creates both the model and model endpoint.
        # TODO: If we just want to perform batch transform, how can we skip creating the endpoint?
        # See https://github.com/aws/sagemaker-python-sdk/issues/49
        predictor = model.deploy(initial_instance_count=1, instance_type=sm_instance_type)

