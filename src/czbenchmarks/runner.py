import os
import pathlib
import tempfile
import boto3
import base64
from typing import Any, List, Union, Optional, Dict

import docker
import yaml
from omegaconf import OmegaConf
import logging

from .constants import (
    INPUT_DATA_PATH_DOCKER,
    MODEL_WEIGHTS_CACHE_PATH,
    MODEL_WEIGHTS_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    RAW_INPUT_DIR_PATH_DOCKER,
    get_numbered_path,
)
from .datasets import BaseDataset
from .models.types import ModelType

logger = logging.getLogger(__name__)


class ContainerRunner:
    """
    Handles Docker container execution logic for running models
    in isolated environments
    """

    def __init__(
        self,
        model_name: Union[str, ModelType],
        gpu: bool = False,
        interactive: bool = False,
        app_mount_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Initialize the ContainerRunner.

        Args:
            model_name: Name of model from models.yaml config or ModelType enum
            gpu: Whether to use GPU acceleration for model execution
            interactive: Whether to run in interactive mode with a bash shell
            app_mount_dir: Optional directory to mount to /app (this will override the
                default /app mount from the docker build!)
            environment: Dictionary of environment variables to pass to the container
            kwargs: Additional arguments to pass to the container as CLI params
        """
        self.client = docker.from_env()

        # Load models config from the default location
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "conf",
            "models.yaml",
        )

        with open(default_config_path) as f:
            cfg = OmegaConf.create(yaml.safe_load(f))

        # Convert string model name to ModelType enum for dataset compatibility
        if isinstance(model_name, str):
            model_type = ModelType[model_name.upper()]
        else:
            model_type = model_name

        # Use model type name for config lookup
        model_key = model_type.name
        if model_key not in cfg.models:
            raise ValueError(f"Model {model_key} not found in models.yaml config")

        model_info = cfg.models[model_key]
        self.image = model_info.model_image_uri

        # Add ECR authentication if the image is from ECR
        if ".dkr.ecr." in self.image:
            try:
                # Extract region from image URI
                region = self.image.split(".")[3]
                registry = self.image.split("/")[0]

                # Get ECR login token
                ecr_client = boto3.client("ecr", region_name=region)
                token = ecr_client.get_authorization_token(
                    registryIds=[registry.split(".")[0]]
                )

                # Decode the authorization token
                auth_data = token["authorizationData"][0]
                token_bytes = base64.b64decode(auth_data["authorizationToken"])
                username, password = token_bytes.decode("utf-8").split(":")

                # Login to Docker with decoded credentials
                self.client.login(
                    username=username,
                    password=password,
                    registry=auth_data["proxyEndpoint"],
                )
            except Exception as e:
                raise RuntimeError(f"Failed to authenticate with ECR: {str(e)}")

        self.model_type = model_type
        self.app_mount_dir = app_mount_dir

        self.gpu = gpu
        self.interactive = interactive

        self.cli_args = kwargs
        self.environment = environment or {}  # Store environment variables

    def run(
        self, datasets: Union[BaseDataset, List[BaseDataset]]
    ) -> Union[BaseDataset, List[BaseDataset]]:
        """Run the model on one or more datasets.

        Args:
            datasets: A single dataset or list of datasets to process

        Returns:
            The processed dataset(s) with model outputs attached
        """
        # Convert single dataset to list for consistent handling
        if not isinstance(datasets, list):
            datasets = [datasets]
            return_single = True
        else:
            return_single = False

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            input_dir_docker = str(pathlib.Path(INPUT_DATA_PATH_DOCKER).parent)
            output_dir_docker = str(pathlib.Path(OUTPUT_DATA_PATH_DOCKER).parent)

            # Store original paths and outputs for restoration after processing
            orig_paths = [os.path.expanduser(d.path) for d in datasets]
            orig_parent_dirs = {
                str(pathlib.Path(p).absolute().parent) for p in orig_paths
            }
            orig_outputs = [d.outputs for d in datasets]

            # Update dataset paths to Docker paths and serialize to temp directory
            for i, dataset in enumerate(datasets):
                # Clear all outputs temporarily for fast serialization
                dataset._outputs = {}
                dataset.path = os.path.join(
                    RAW_INPUT_DIR_PATH_DOCKER, pathlib.Path(orig_paths[i]).name
                )
                dataset.unload_data()  # Free memory before serialization
                input_path = get_numbered_path(
                    os.path.join(input_dir, pathlib.Path(INPUT_DATA_PATH_DOCKER).name),
                    i,
                )
                dataset.serialize(input_path)

            # Setup Docker volume mounts
            volumes = {
                input_dir: {"bind": input_dir_docker, "mode": "ro"},
                output_dir: {"bind": output_dir_docker, "mode": "rw"},
                self._get_weights_cache_path(): {
                    "bind": MODEL_WEIGHTS_PATH_DOCKER,
                    "mode": "rw",
                },
            }

            # Mount original dataset directories as read-only volumes
            for parent_dir in orig_parent_dirs:
                volumes[parent_dir] = {
                    "bind": RAW_INPUT_DIR_PATH_DOCKER,
                    "mode": "ro",
                }

            # Add app mount if specified
            if self.app_mount_dir:
                expanded_path = os.path.abspath(os.path.expanduser(self.app_mount_dir))
                if not os.path.exists(expanded_path):
                    raise ValueError(f"Mount directory does not exist: {expanded_path}")
                volumes[expanded_path] = {
                    "bind": "/app",
                    "mode": "rw",
                }

            # Run container and process results
            try:
                self._run_container(volumes)

                # Load and process results
                result_datasets = []
                for i, _ in enumerate(datasets):
                    output_path = get_numbered_path(
                        os.path.join(
                            output_dir,
                            pathlib.Path(OUTPUT_DATA_PATH_DOCKER).name,
                        ),
                        i,
                    )
                    dataset = BaseDataset.deserialize(output_path)
                    dataset.path = orig_paths[i]  # Restore original path

                    # Get new outputs from container
                    new_outputs = dataset.outputs

                    # Restore original outputs but exclude current model
                    dataset._outputs = orig_outputs[i]

                    # Add new outputs from container
                    dataset.outputs[self.model_type] = new_outputs[self.model_type]

                    dataset.load_data()  # Load the dataset back into memory
                    result_datasets.append(dataset)

                return result_datasets[0] if return_single else result_datasets

            except Exception as e:
                # Restore original paths and outputs on error
                for dataset, orig_path, orig_output in zip(
                    datasets, orig_paths, orig_outputs
                ):
                    dataset.path = orig_path
                    dataset._outputs = orig_output
                raise e

    def _run_container(self, volumes: dict):
        """Run the Docker container with the specified volumes.

        Args:
            volumes: Dictionary mapping host paths to container mount points

        Raises:
            RuntimeError: If the container exits with a non-zero status code
        """
        image_name = self.image.split("/")[-1].split(":")[0]
        model_weights_cache_path = os.path.expanduser(
            os.path.join(MODEL_WEIGHTS_CACHE_PATH, image_name)
        )
        os.makedirs(model_weights_cache_path, exist_ok=True)

        # Pull the image first if it's from ECR
        if ".dkr.ecr." in self.image:
            try:
                logger.info(f"Pulling image {self.image}...")
                self.client.images.pull(self.image, platform="linux/amd64")
            except Exception as e:
                raise RuntimeError(f"Failed to pull image {self.image}: {str(e)}")

        # Prepare command based on mode (interactive or CLI args)
        command = None
        if self.cli_args and not self.interactive:
            command = []
            for key, value in self.cli_args.items():
                command.extend([f"--{key}", str(value)])

        # Add platform specification for ECR images
        platform = "linux/amd64" if ".dkr.ecr." in self.image else None

        # Create the container with appropriate configuration
        container = self.client.containers.create(
            image=self.image,
            command=command,
            volumes=volumes,
            environment=self.environment,
            runtime="nvidia" if self.gpu else None,
            tty=self.interactive,
            stdin_open=self.interactive,
            entrypoint=(["/bin/bash"] if self.interactive else None),
            platform=platform,
        )

        try:
            container.start()
            # Stream logs in real-time for monitoring
            for log in container.logs(stream=True, follow=True):
                logger.info(log.decode().strip())

            # Wait for container to finish and check exit status
            result = container.wait()
            if result["StatusCode"] != 0:
                raise RuntimeError(
                    f"Container exited with status code {result['StatusCode']}"
                )

        except Exception as e:
            raise e

        finally:
            container.remove()  # Clean up container regardless of outcome

    def _get_weights_cache_path(self):
        """Get the path to the model weights cache directory.

        Returns:
            Path to the model-specific weights cache directory
        """
        image_name = self.image.split("/")[-1].split(":")[0]
        return os.path.expanduser(os.path.join(MODEL_WEIGHTS_CACHE_PATH, image_name))


def run_inference(
    model_name: str,
    dataset: BaseDataset,
    gpu: bool = True,
    interactive: bool = False,
    app_mount_dir: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    **kwargs,
) -> BaseDataset:
    """Convenience function to run inference on a single dataset.

    Args:
        model_name: Name of the model to run
        dataset: Dataset to process
        gpu: Whether to use GPU acceleration
        interactive: Whether to run in interactive mode
        app_mount_dir: Optional directory to mount to /app in the container
            (this will override the default /app mount from the docker build!)
        environment: Dictionary of environment variables to pass to the container
        **kwargs: Additional arguments to pass to the container as CLI params

    Returns:
        The processed dataset with model outputs attached
    """
    runner = ContainerRunner(
        model_name,
        gpu,
        interactive=interactive,
        app_mount_dir=app_mount_dir,
        environment=environment,
        **kwargs,
    )
    return runner.run(dataset)
