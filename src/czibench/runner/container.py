import os
import tempfile
import docker
import pathlib
from typing import Any
from ..constants import INPUT_DATA_PATH_DOCKER, OUTPUT_DATA_PATH_DOCKER
from ..datasets.base import BaseDataset

# TODO: Need a manifest to document model names and their image names
class ContainerRunner:
    """Handles Docker container execution logic"""
    
    def __init__(
        self,
        image: str,
        gpu: bool = False,
        **kwargs: Any
    ):
        self.image = image
        self.gpu = gpu
        self.cli_args = kwargs
        self.client = docker.from_env()

    def run(self, data: BaseDataset) -> BaseDataset:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup I/O paths
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            input_dir = str(pathlib.Path(INPUT_DATA_PATH_DOCKER).parent)
            output_dir = str(pathlib.Path(OUTPUT_DATA_PATH_DOCKER).parent)
            input_filename = str(pathlib.Path(INPUT_DATA_PATH_DOCKER).name)
            output_filename = str(pathlib.Path(OUTPUT_DATA_PATH_DOCKER).name)
            
            input_path = os.path.join(input_dir, input_filename)
            output_path = os.path.join(output_dir, output_filename)
            data.save(input_path)
            
            volumes = {
                input_dir: {"bind": input_dir, "mode": "ro"},
                output_dir: {"bind": output_dir, "mode": "rw"}
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
                remove=True
            )
            
            return BaseDataset.load(output_path)