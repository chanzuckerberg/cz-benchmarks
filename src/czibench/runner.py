import os
import tempfile
import docker
import pathlib
from typing import Any, Union, List

from .constants import (
    INPUT_DATA_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    MODEL_WEIGHTS_PATH_DOCKER,
    RAW_INPUT_DIR_PATH_DOCKER,
    MODEL_WEIGHTS_CACHE_PATH,
    get_numbered_path,
)

from .datasets.base import BaseDataset


class ContainerRunner:
    """Handles Docker container execution logic"""

    def __init__(self, image: str, gpu: bool = False, **kwargs: Any):
        self.image = image
        self.gpu = gpu
        self.cli_args = kwargs
        self.client = docker.from_env()

    def run(
        self, datasets: Union[BaseDataset, List[BaseDataset]]
    ) -> Union[BaseDataset, List[BaseDataset]]:
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

            # Store original paths
            orig_paths = [os.path.expanduser(d.path) for d in datasets]
            orig_parent_dirs = {str(pathlib.Path(p).parent) for p in orig_paths}

            # Update dataset paths and serialize
            for i, dataset in enumerate(datasets):
                dataset.path = os.path.join(
                    RAW_INPUT_DIR_PATH_DOCKER, pathlib.Path(orig_paths[i]).name
                )
                dataset.unload_data()
                input_path = get_numbered_path(
                    os.path.join(input_dir, pathlib.Path(INPUT_DATA_PATH_DOCKER).name), i
                )
                dataset.serialize(input_path)

            # Setup volumes including all parent directories
            volumes = {
                input_dir: {"bind": input_dir_docker, "mode": "ro"},
                output_dir: {"bind": output_dir_docker, "mode": "rw"},
                self._get_weights_cache_path(): {
                    "bind": MODEL_WEIGHTS_PATH_DOCKER,
                    "mode": "rw",
                },
            }

            # Add all unique parent directories as volumes
            for parent_dir in orig_parent_dirs:
                volumes[parent_dir] = {"bind": RAW_INPUT_DIR_PATH_DOCKER, "mode": "ro"}

            # Run container and process results
            try:
                self._run_container(volumes)

                # Load results
                result_datasets = []
                for i in range(len(datasets)):
                    output_path = get_numbered_path(
                        os.path.join(
                            output_dir, pathlib.Path(OUTPUT_DATA_PATH_DOCKER).name
                        ),
                        i
                    )
                    dataset = BaseDataset.deserialize(output_path)
                    dataset.path = orig_paths[i]
                    dataset.load_data()
                    result_datasets.append(dataset)

                return result_datasets[0] if return_single else result_datasets

            except Exception as e:
                # Restore original paths on error
                for dataset, orig_path in zip(datasets, orig_paths):
                    dataset.path = orig_path
                raise e

    def _run_container(self, volumes: dict):
        image_name = self.image.split("/")[-1].split(":")[0]
        model_weights_cache_path = os.path.expanduser(
            os.path.join(MODEL_WEIGHTS_CACHE_PATH, image_name)
        )
        os.makedirs(model_weights_cache_path, exist_ok=True)

        command = []
        if self.cli_args:
            for key, value in self.cli_args.items():
                command.extend([f"--{key}", str(value)])

        container = self.client.containers.create(
            image=self.image,
            command=command,
            volumes=volumes,
            runtime="nvidia" if self.gpu else None,
        )

        try:
            container.start()
            # Stream logs in real-time
            for log in container.logs(stream=True, follow=True):
                print(log.decode().strip())

            # Wait for container to finish
            result = container.wait()
            if result["StatusCode"] != 0:
                raise RuntimeError(
                    f"Container exited with status code {result['StatusCode']}"
                )

        except Exception as e:
            raise e

        finally:
            container.remove()

    def _get_weights_cache_path(self):
        image_name = self.image.split("/")[-1].split(":")[0]
        return os.path.expanduser(os.path.join(MODEL_WEIGHTS_CACHE_PATH, image_name))
