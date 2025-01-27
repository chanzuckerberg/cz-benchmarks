import os
import tempfile
import docker
import pathlib
import shutil
from typing import Any
from .constants import INPUT_DATA_PATH_DOCKER, OUTPUT_DATA_PATH_DOCKER, ARTIFACTS_PATH_DOCKER
from .datasets.base import BaseDataset

class ContainerRunner:
    """Handles Docker container execution logic"""
    
    def __init__(
        self,
        image: str,
        gpu: bool = False,
        artifact_mount_path: str = "/mnt/efs",
        **kwargs: Any
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
            data.path = os.path.join(input_dir_docker, pathlib.Path(orig_path).name)
            # FIXME: Can we mount the parent directory instead of the file?
            # (Alec): I ran into permission issues when trying to mount the parent directory hence this workaround
            shutil.copy2(orig_path, f"{input_dir}/{pathlib.Path(orig_path).name}")            
            data.serialize(input_path)

            volumes = {
                input_dir: {"bind": input_dir_docker, "mode": "ro"},
                output_dir: {"bind": output_dir_docker, "mode": "rw"},
                self.artifact_mount_path: {"bind": ARTIFACTS_PATH_DOCKER, "mode": "rw"},
            }
            
            command = []
            if self.cli_args:
                for key, value in self.cli_args.items():
                    command.extend([f"--{key}", str(value)])
            
            # Run container
            self.client.containers.run(
                image=self.image,
                command=command,
                volumes=volumes,
                runtime="nvidia" if self.gpu else None,
                remove=True,
            )
            
            data = BaseDataset.deserialize(output_path)
            data.path = orig_path
            data.load_data()
            return data
