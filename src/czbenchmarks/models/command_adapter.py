import os
import shlex
import subprocess
import logging
from typing import Dict, Any

from . import ModelAdapter

logger = logging.getLogger(__name__)

class ShellAdapter(ModelAdapter):
    """
    Adapter to execute shell commands for model inference in case a command-line tool is available for model execution.
    """

    def __init__(self, command_template: str, config: Dict[str, Any] = None):
        """
        Initialize the ShellAdapter with the shell command template and configuration.
        """
        if config is None:
            config = {}
        config["command_template"] = command_template
        super().__init__(config)

    def _validate_config(self):
        if "command_template" not in self.config:
            raise ValueError("Config must include 'command_template'.")
        self.timeout = self.config.get("timeout", 3600)
        self.stream_output = self.config.get("stream_output", True)

    def run(self, input_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, str]:
        command_template = self.config.get("command_template")

        if not command_template:
            raise ValueError("Command template must be specified in config.")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Prepare placeholder mapping
        mapping = {"input": input_path, "output": output_path, **params}

        try:
            # Format the command template with placeholders
            command = command_template.format(**mapping)
        except KeyError as e:
            raise ValueError(f"Failed to format command template: missing key {e}")

        logger.info(f"Executing command: {command}")

        try:
            # Execute the command
            process = subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
            )

            stdout, stderr = process.communicate(timeout=self.timeout)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Command exited with code {process.returncode}.\nSTDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
                )

            # Save stdout and stderr to files if not streaming
            artifacts = {}
            if not self.stream_output:
                stdout_file = os.path.join(output_path, "stdout.txt")
                stderr_file = os.path.join(output_path, "stderr.txt")
                with open(stdout_file, "w") as f:
                    f.write(stdout.decode())
                with open(stderr_file, "w") as f:
                    f.write(stderr.decode())
                artifacts["stdout"] = stdout_file
                artifacts["stderr"] = stderr_file

            # Capture any files created under output_path
            for root, _, files in os.walk(output_path):
                for fname in files:
                    rel_path = os.path.relpath(os.path.join(root, fname), output_path)
                    artifacts.setdefault(rel_path, os.path.join(root, fname))

            return {"status": "success", "artifacts": artifacts}

        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(f"Command timed out after {self.timeout} seconds.")
        except Exception as exc:
            logger.exception("Error while executing shell command.")
            raise RuntimeError(f"Error while executing shell command: {exc}") from exc