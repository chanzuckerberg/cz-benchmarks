import os
import docker
import logging
from typing import Dict, Any
from . import ModelAdapter

logger = logging.getLogger(__name__)

class DockerAdapter(ModelAdapter):
    """
    Adapter to execute a Docker image for model inference.
    """

    def __init__(self, image_name: str, config: Dict[str, Any] = None):
        """
        Initialize the DockerAdapter with the Docker image name and configuration.
        """
        if config is None:
            config = {}
        config["image_name"] = image_name
        super().__init__(config)
        self.client = docker.from_env()

    def _validate_config(self):
        if "image_name" not in self.config:
            raise ValueError("Config must include 'image_name'.")

    def _setup(self):
        pass

    def run(self, input_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, str]:
        image_name = self.config.get("image_name")

        if not image_name:
            raise ValueError("Docker image name must be specified in config.")

        try:
            volumes = {
                os.path.abspath(input_path): {"bind": "/input", "mode": "ro"},
                os.path.abspath(output_path): {"bind": "/output", "mode": "rw"},
            }

            # Prepare command arguments for the Docker container
            command = []
            for key, value in params.items():
                command.extend([f"--{key}", str(value)])

            logger.info(f"Running Docker container with image: {image_name}")
            container = self.client.containers.run(
                image=image_name,
                command=command,
                volumes=volumes,
                detach=True,
            )

            # Stream logs for monitoring
            for log in container.logs(stream=True, follow=True):
                logger.info(log.decode().strip())

            # Wait for container to finish and check exit status
            result = container.wait()
            if result["StatusCode"] != 0:
                raise RuntimeError(f"Container exited with status code {result['StatusCode']}")

            return {"status": "success", "result": f"Output saved to {output_path}"}

        except Exception as exc:
            logger.exception("Error while running Docker container")
            raise RuntimeError(f"Error while running Docker container: {exc}") from exc

        finally:
            if 'container' in locals():
                container.remove()  # Clean up container

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a Docker image using DockerAdapter.")
    parser.add_argument("--image", required=True, help="Name of the Docker image.")
    parser.add_argument("--input", required=True, help="Input path to mount in the container.")
    parser.add_argument("--output", required=True, help="Output path to mount in the container.")
    parser.add_argument("--params", default="{}", help="JSON string of additional parameters.")

    args = parser.parse_args()

    import json
    params = json.loads(args.params)

    adapter = DockerAdapter(args.image)
    result = adapter.run(args.input, args.output, params)
    print(result)