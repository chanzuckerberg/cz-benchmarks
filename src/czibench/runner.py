import os
import pathlib
import tempfile
from typing import Any

import docker

from .constants import (
    ARTIFACTS_PATH_DOCKER,
    INPUT_DATA_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    RAW_INPUT_DIR_PATH_DOCKER,
)
from .datasets.base import BaseDataset


class ContainerRunner:
    """Handles Docker container execution logic"""

    def __init__(
        self,
        image: str,
        gpu: bool = False,
        artifact_mount_path: str = "/mnt/efs",
        **kwargs: Any,
    ):
        self.image = image
        self.gpu = gpu
        self.cli_args = kwargs
        self.artifact_mount_path = artifact_mount_path
        self.client = docker.from_env()

    def run(self, data: BaseDataset) -> BaseDataset:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup I/O paths
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            input_dir_docker = str(pathlib.Path(INPUT_DATA_PATH_DOCKER).parent)
            output_dir_docker = str(pathlib.Path(OUTPUT_DATA_PATH_DOCKER).parent)
            input_filename = str(pathlib.Path(INPUT_DATA_PATH_DOCKER).name)
            output_filename = str(pathlib.Path(OUTPUT_DATA_PATH_DOCKER).name)

            input_path = os.path.join(input_dir, input_filename)
            output_path = os.path.join(output_dir, output_filename)

            orig_path = os.path.expanduser(data.path)
            data.path = os.path.join(
                RAW_INPUT_DIR_PATH_DOCKER, pathlib.Path(orig_path).name
            )

            orig_parent_dir = str(pathlib.Path(orig_path).parent)

            data.unload_data()
            data.serialize(input_path)

            volumes = {
                input_dir: {"bind": input_dir_docker, "mode": "ro"},
                output_dir: {"bind": output_dir_docker, "mode": "rw"},
                self.artifact_mount_path: {
                    "bind": ARTIFACTS_PATH_DOCKER,
                    "mode": "rw",
                },
                orig_parent_dir: {
                    "bind": RAW_INPUT_DIR_PATH_DOCKER,
                    "mode": "ro",
                },
            }

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
                        "Container exited with status code" f" {result['StatusCode']}"
                    )

                data = BaseDataset.deserialize(output_path)
                data.path = orig_path
                data.load_data()

            except Exception as e:
                data.path = orig_path
                raise e

            finally:
                container.remove()

            return data
