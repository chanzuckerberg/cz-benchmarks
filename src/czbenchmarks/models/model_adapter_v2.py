import os
import sys
import json
import logging
import yaml
import abc
from typing import Dict, Any, List, Optional
import subprocess

class ModelAdapter(abc.ABC):
    """
    Abstract base class for inference adapters, with full lifecycle management.

    Config schema:
      resource:                # Resource requirements for running the model
        use_gpu: bool
        gpu_memory: int        # in MB
        cpu_cores: int
        cpu_memory: int        # in MB
        disk_space: int        # in MB

      model_validation:        # Validation rules for model inputs
        supported_modalities: List[str]      # e.g. ["image", "text"]
        supported_filetypes: List[str]       # e.g. [".jpg", ".png", ".txt"]
        input_schema: Dict[str,Any] (optional) # e.g. {"required_fields": ["field1", "field2"]}

      input:                # Input data type, e.g. "src/czbenchmarks/datasets/base.py" or "Any" or FilePath

      model_params:            # Model-specific parameters
        batch_size: int
        random_seed: int
        weights: "path/to/model_weights"
        # ... other model parameters

      output:                  # Output configuration
        model_output_path: str
        log_path: str (optional)

    Lifecycle:
      1) _validate_inputs() # validate input file type, input datatype, and input schema
      2) _setup()
      3) _preprocess()
      4) _run()
      5) _postprocess()
      6) _cleanup()

    Provides run() orchestrator.
    """

    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_handler)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None
    ):
        self.config = self.load_config(config, config_file)
        self._validate_config_schema(self.config)
        self.resource = self.config.get('resource', {})
        self.model_validation = self.config.get('model_validation', {})
        self.input_type = self.config.get('input', None)
        self.model_params = self.config.get('model_params', {})
        self.output_config = self.config.get('output', {})
        self.inputs = self.config.get('inputs', {})

    # We can enhance to load config from command line args or environment variables, if desired.
    @staticmethod
    def load_config(
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if config_file:
            ext = os.path.splitext(config_file)[1].lower()
            with open(config_file, 'r') as f:
                if ext in ('.yaml', '.yml'):
                    merged.update(yaml.safe_load(f))
                elif ext == '.json':
                    merged.update(json.load(f))
                else:
                    raise ValueError(f"Unsupported config extension: {ext}")
        if config:
            merged.update(config)
        return merged

    @staticmethod
    def _validate_config_schema(config: Dict[str, Any]) -> None:
        resource = config.get('resource', {})
        if not isinstance(resource, dict):
            raise ValueError("Config 'resource' section must be a dict.")
        for key, typ in [
            ('use_gpu', bool),
            ('gpu_memory', int),
            ('cpu_cores', int),
            ('cpu_memory', int),
            ('disk_space', int)
        ]:
            if key not in resource:
                raise ValueError(f"Config 'resource' missing required key: '{key}'")
            if not isinstance(resource[key], typ):
                raise ValueError(f"Config 'resource.{key}' must be of type {typ.__name__}")

        if 'input' not in config:
            raise ValueError("Config missing required key: 'input'")

        model_params = config.get('model_params', {})
        if not isinstance(model_params, dict):
            raise ValueError("Config 'model_params' section must be a dict.")
        for key, typ in [
            ('batch_size', int),
            ('random_seed', int),
            ('weights', str)
        ]:
            if key not in model_params:
                raise ValueError(f"Config 'model_params' missing required key: '{key}'")
            if not isinstance(model_params[key], typ):
                raise ValueError(f"Config 'model_params.{key}' must be of type {typ.__name__}")

        output = config.get('output', {})
        if not isinstance(output, dict):
            raise ValueError("Config 'output' section must be a dict.")
        if 'model_output_path' not in output or not isinstance(output['model_output_path'], str):
            raise ValueError("Config 'output.model_output_path' must be a string.")
        if 'log_path' in output and not isinstance(output['log_path'], str):
            raise ValueError("Config 'output.log_path' must be a string if provided.")

        model_validation = config.get('model_validation', {})
        if not isinstance(model_validation, dict):
            raise ValueError("Config 'model_validation' section must be a dict.")
        if 'supported_modalities' not in model_validation or not isinstance(model_validation['supported_modalities'], list):
            raise ValueError("Config 'model_validation.supported_modalities' must be a list of strings.")
        if 'supported_filetypes' not in model_validation or not isinstance(model_validation['supported_filetypes'], list):
            raise ValueError("Config 'model_validation.supported_filetypes' must be a list of strings.")
        if 'input_schema' in model_validation and not isinstance(model_validation['input_schema'], dict):
            raise ValueError("Config 'model_validation.input_schema' must be a dict if provided.")

    def supported_modalities(self) -> List[str]:
        return self.model_validation.get('supported_modalities', [])

    def supported_filetypes(self) -> List[str]:
        return self.model_validation.get('supported_filetypes', [])

    def input_schema(self) -> Dict[str, Any]:
        return self.model_validation.get('input_schema', {})

    def _validate_inputs(self) -> None:
        modalities = self.supported_modalities()
        filetypes = self.supported_filetypes()
        schema = self.input_schema()
        input_type = self.input_type
        inputs = self.inputs

        for mod, value in inputs.items():
            if modalities and mod not in modalities:
                raise ValueError(f"Unsupported modality '{mod}'. Supported: {modalities}")

            if (input_type and "FilePath" in str(input_type)) or filetypes:
                if not isinstance(value, str) or not os.path.exists(value):
                    raise FileNotFoundError(f"Input for '{mod}' not found: {value}")
                ext = os.path.splitext(value)[1].lower()
                if filetypes and ext not in filetypes:
                    raise ValueError(f"Modality '{mod}' expects filetypes {filetypes}, got '{ext}'")

            if schema:
                req_fields = schema.get('required_fields', [])
                if req_fields:
                    if isinstance(value, dict):
                        missing = [f for f in req_fields if f not in value]
                        if missing:
                            raise ValueError(f"Missing required fields {missing} for modality '{mod}'")
                    else:
                        self.logger.debug(f"Schema validation skipped for non-dict input for modality '{mod}'")

    def _setup(self) -> None:
        """Hook: prepare resources (e.g. clone repo, create venv)."""
        pass

    def _preprocess(self) -> None:
        """Hook before inference (e.g. data staging)."""
        pass

    def _postprocess(self) -> None:
        """Hook after inference (e.g. saving logs)."""
        pass

    def _cleanup(self) -> None:
        """Hook to tear down resources (e.g. temp dirs)."""
        pass

    @abc.abstractmethod
    def _run(self) -> Any:
        """
        Core inference logic. Return artifact keys -> file paths.
        Subclasses must implement this method.
        """
        pass

    def run(self) -> Any:
        """
        Execute the full adapter lifecycle.
        Subclasses should only call this method to run the pipeline.
        """
        self.logger.info("=== Starting inference pipeline ===")
        try:
            self._validate_inputs()
            self._setup()
            self._preprocess()
            result = self._run()
            self._postprocess()
            return result
        finally:
            self._cleanup()



#-----------------------------------------------------------------------------
# DockerAdapter implementation
#-----------------------------------------------------------------------------

class DockerAdapter(ModelAdapter):
    """
    Run a Docker image with multi-modal inputs.

    Config keys (in addition to ModelAdapter):
      docker:
        image: str
        command_template: str  # placeholders: inputs.<modality>, output, params
        timeout: int (optional)
        extra_args: List[str] (optional)
      model_validation:
        supported_modalities: List[str]
        supported_filetypes: List[str] 
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        super().__init__(config, config_file)
        self.docker_config = self.config.get('docker', {})
        self._validate_docker_config(self.docker_config)
        self.timeout = self.docker_config.get('timeout', None)
        self.extra_args = self.docker_config.get('extra_args', [])

    @staticmethod
    def _validate_docker_config(docker_config: Dict[str, Any]) -> None:
        for k in ('image', 'command_template'):
            if k not in docker_config:
                raise ValueError(f"Docker config missing required key: '{k}'")
        if not isinstance(docker_config['image'], str):
            raise ValueError("Docker config 'image' must be a string.")
        if not isinstance(docker_config['command_template'], str):
            raise ValueError("Docker config 'command_template' must be a string.")

    def _preprocess(self) -> None:
        # Validate input files and modalities
        modalities = self.supported_modalities()
        filetypes = self.supported_filetypes()
        inputs = self.inputs

        for mod, value in inputs.items():
            if modalities and mod not in modalities:
                raise ValueError(f"Unsupported modality '{mod}'. Supported: {modalities}")
            if not isinstance(value, str) or not os.path.exists(value):
                raise FileNotFoundError(f"Input for '{mod}' not found: {value}")
            ext = os.path.splitext(value)[1].lower()
            # filetypes can be a dict or a list
            if isinstance(filetypes, dict):
                allowed = filetypes.get(mod, [])
            else:
                allowed = filetypes
            if allowed and ext not in allowed:
                raise ValueError(f"Modality '{mod}' expects filetypes {allowed}, got '{ext}'")

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_config['model_output_path'])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _run(self) -> Dict[str, str]:
        """
        Run the docker container with the configured command.
        Returns a dict with output artifact paths.
        """
        inputs = self.inputs
        output_path = self.output_config['model_output_path']
        params = self.model_params
        docker_image = self.docker_config['image']
        command_template = self.docker_config['command_template']
        timeout = self.timeout

        # Prepare Docker volume mounts and command placeholders
        mounts: List[str] = []
        placeholders: Dict[str, Any] = {}

        for mod, path in inputs.items():
            container_path = f"/inputs/{mod}"
            mounts += ['-v', f"{os.path.abspath(path)}:{container_path}:ro"]
            placeholders[f"inputs.{mod}"] = container_path

        output_container_path = "/output"
        mounts += ['-v', f"{os.path.abspath(output_path)}:{output_container_path}:rw"]
        placeholders['output'] = output_container_path

        # Add model_params as placeholders
        placeholders.update(params)

        # Format command string
        try:
            cmd_str = command_template.format(**placeholders)
        except KeyError as e:
            raise ValueError(f"Missing placeholder in command_template: {e}")

        docker_cmd = (
            ['docker', 'run', '--rm'] +
            mounts +
            (self.extra_args if self.extra_args else []) +
            [docker_image] +
            cmd_str.split()
        )

        self.logger.info("Running DockerAdapter: %s", " ".join(docker_cmd))
        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"Docker command timed out after {timeout} seconds")
        if proc.returncode:
            self.logger.error(out)
            raise RuntimeError(f"Docker exited with code {proc.returncode}: {out}")
        self.logger.info("Docker run completed successfully.")
        return {'model_output_path': output_path}