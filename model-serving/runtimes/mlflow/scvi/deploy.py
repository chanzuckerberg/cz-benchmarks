import argparse
import shutil
from datetime import datetime
from typing import Tuple

import boto3
import mlflow
import mlflow.models
import mlflow.pyfunc
import numpy as np
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from sagemaker.serve import ModelBuilder, SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.async_inference import AsyncInferenceConfig

from mlflow_scvi import MLflowSCVI

REGION = "us-west-2"
S3_BUCKET = "omar-data"


def save_mlflow_model_locally():   
    """Save the MLflow model locally."""
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
        
def create_sagemaker_model(model_name: str, sm_deploy_mode: str, sm_instance_type: str):# -> Model:
    """Create a SageMaker model and deploy it to an endpoint."""
    model_schema = SchemaBuilder(
        sample_input="s3://path/input.h5ad",
        sample_output=np.array([[0.0020090353209525347, 0.000620177364908158]])
    )
    
    # Deploy MLflow model with ModelBuilder
    # https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments-model-deployment.html
    # Note: Requires AWS_PROFILE or AWS_DEFAULT_REGION environment variable is set
    # TODO: How do we specify async endpoint?
    model_builder = ModelBuilder(
        name=model_name,
        mode=sm_deploy_mode,
        # Note: To ensure use of GPU Docker base image, must specify instance_type
        instance_type=sm_instance_type,
        # FIXME: A schema is required to avoid failing with "cannot serialize" error; 
        # however, the model is actually being invoked on deploy with the schema's sample inputs--why!?
        schema_builder=model_schema,
        # TODO: This role is assumed to have been created already by having run model-serving/runtimes/sagemaker/scvi/remote/deploy.py (remote async sagemaker endpoint)
        role_arn="arn:aws:iam::058264139299:role/OmarSageMakerRole",
        model_metadata={
            # both model path and tracking server ARN are required if you use an mlflow run ID or mlflow model registry path as input
            "MLFLOW_MODEL_PATH": mlflow_model_path,
            # "MLFLOW_TRACKING_ARN": "arn:aws:sagemaker:region:account-id:mlflow-tracking-server/tracking-server-name"
        }
    )
    # TODO: The `build` step takes many minutes!  Why?
    model = model_builder.build() # this uploads the *.tar.gz file
    model.create() # this creates the model entity in SageMaker
    
    return model

def find_existing_sagemaker_model(sm_client, model_name_prefix: str) -> Tuple[str, str]:
    """
    Find the latest model in SageMaker. This is an optional step, to support
    just creating an endpoint for an existing model, without having to re-create the model.
    The finds latest model in SageMaker with a name matching the provided prefix,
    to obtain the model name (with timestamp).
    """
    models = sm_client.list_models(
        NameContains=model_name_prefix,
        SortOrder="Descending",
        SortBy="CreationTime",
        MaxResults=1,
    )
    if not models["Models"]:
        print("No latest model in SageMaker. Aborting.")
        exit(1)
        
    model_name = models["Models"][0]["ModelName"]
    timestamp = model_name.split("-")[-1]
    print(f"Found model {model_name} in SageMaker.")
        
    return model_name, timestamp
        

def register_sagemaker_model(sm_client, model, sm_instance_type: str):
    """Register the model in the SageMaker model registry as versioned model. This is an optional step, not required for deployment."""
    model_package_name = "scvi-model-package-group"
    response = sm_client.describe_model_package_group(
        ModelPackageGroupName=model_package_name
    )
    if not response["ModelPackageGroupArn"]:
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=model_package_name
    )
    
    model.register(
        model_package_group_name=model_package_name,
        inference_instances=[sm_instance_type],
        approval_status="Approved"
    )


def create_sagemaker_endpoint(sm_client, model_name: str , timestamp: str, sm_instance_type: str):
    async_config = AsyncInferenceConfig(
        output_path=f"s3://{S3_BUCKET}/scvi-async-output/",
        failure_path=f"s3://{S3_BUCKET}/scvi-async-failure/",
        # TODO: Add notification configs for success and failure
    )

    response = sm_client.create_endpoint_config(
        EndpointConfigName=model_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "VariantName": "SCVIHumanMouseVariant",
                "InitialInstanceCount": 1,
                "InstanceType": sm_instance_type,
            }
        ],
        AsyncInferenceConfig=async_config._to_request_dict(),
        # ExecutionRoleArn??
    )

    # # Real-time endpoint
    # response = sm_client.create_endpoint_config(
    #     EndpointConfigName=model_name,
    #     ProductionVariants=[
    #         {
    #             "VariantName": "AllTraffic",
    #             "ModelName": model_name,
    #             "InstanceType": sm_instance_type,
    #             "InitialInstanceCount": 1,
    #             "InitialVariantWeight": 1.0,
    #         #     "AutoScalingPolicy": {
    #         #         "InitialInstanceCount": 1,
    #         #         "InitialVariantWeight": 1.0,
    #         #         "MinCapacity": 0,
    #         #         "MaxCapacity": 1,
    #         #         "ScalingPolicyName": "ScaleToZeroPolicy",
    #         #         "ScalingPolicyType": "TargetTrackingScaling",
    #         #         "TargetTrackingScalingPolicyConfiguration": {
    #         #             "TargetValue": 0.0,
    #         #             "PredefinedMetricSpecification": {
    #         #             "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
    #         #             },
    #         #             "ScaleInCooldown": 300,
    #         #             "ScaleOutCooldown": 300
    #         #         }
    #         #     }
    #         }
    #     ]
    # )

    print(response)
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200, "Failed to create endpoint config"
    print(f"Endpoint config {model_name} created.")
    
    response = sm_client.create_endpoint(
        EndpointName=model_name,
        EndpointConfigName=model_name
    )
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200, "Failed to create endpoint"
    print(f"Endpoint {model_name} created with ARN {response['EndpointArn']}.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save and log MLflow model")
    parser.add_argument("--target", choices=["local", "databricks", "sagemaker", "sagemaker-local"], default="local", required=True, help="Target to save the model: 'local' or 'databricks'")
    parser.add_argument("--experiment-name", type=str, required=False, help="Name of the MLflow experiment")
    parser.add_argument("--initial-sagemaker-step", default="mlflow-package", choices=["mlflow-package", "sagemaker-model-create", "sagemaker-endpoint"], help="The step to start at for SageMaker remote deployments, for debugging purposes")
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
      
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_prefix = "scvi-mlflow"
        model_name = f"{model_name_prefix}-{timestamp}"

        sm_client = boto3.client("sagemaker")

        if args.target.endswith("local"):
            sm_deploy_mode = Mode.LOCAL_CONTAINER
            sm_instance_type = "local"            
        else:
            sm_deploy_mode = Mode.SAGEMAKER_ENDPOINT
            sm_instance_type = "ml.g4dn.xlarge"


        if args.initial_sagemaker_step == "mlflow-package":
            save_mlflow_model_locally()
            
        if args.initial_sagemaker_step in ["mlflow-package", "sagemaker-model-create"]:
            model = create_sagemaker_model(model_name,
                                           sm_deploy_mode,
                                           sm_instance_type)
        else:   
            model_name, time_stamp = find_existing_sagemaker_model(sm_client, model_name_prefix)
            
        if args.target == "sagemaker-local":
            predictor = model.deploy(initial_instance_count=1, instance_type=sm_instance_type)
        else:
            create_sagemaker_endpoint(
                sm_client,
                model_name,
                timestamp,
                sm_instance_type
            )
            
        # # This creates both the model and model endpoint.
        # # TODO: If we just want to perform batch transform, how can we skip creating the endpoint?
        # # See https://github.com/aws/sagemaker-python-sdk/issues/49
        # predictor = model.deploy(initial_instance_count=1, instance_type=sm_instance_type)

