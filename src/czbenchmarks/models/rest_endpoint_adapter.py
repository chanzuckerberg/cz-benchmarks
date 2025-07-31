import os
import json
import logging
import requests
from typing import Dict, Any

from . import ModelAdapter

logger = logging.getLogger(__name__)

class RestEndpointAdapter(ModelAdapter):
    """
    Adapter to execute a REST API endpoint for model inference.
    """

    def __init__(self, endpoint_url: str, config: Dict[str, Any] = None):
        """
        Initialize the RestEndpointAdapter with the endpoint URL and configuration.
        """
        if config is None:
            config = {}
        config["endpoint_url"] = endpoint_url
        super().__init__(config)

    def _validate_config(self):
        if "endpoint_url" not in self.config:
            raise ValueError("Config must include 'endpoint_url'.")
        self.method = self.config.get("method", "POST").upper()
        if self.method not in ("POST", "GET"):
            raise ValueError("RestEndpointAdapter only supports GET or POST methods.")
        self.headers = self.config.get("headers", {})
        self.timeout = self.config.get("timeout", 60)
        self.payload_type = self.config.get("payload_type", "json")
        self.body_template = self.config.get("body_template", None)

    def run(self, input_path: str, output_path: str, params: Dict[str, Any]) -> Dict[str, str]:
        endpoint_url = self.config.get("endpoint_url")

        if not endpoint_url:
            raise ValueError("REST API endpoint URL must be specified in config.")

        os.makedirs(output_path, exist_ok=True)

        # Prepare the request payload
        json_body = None
        data_body = None
        files_body = None

        if self.payload_type.lower() == "json":
            if isinstance(self.body_template, dict):
                try:
                    json_body = json.loads(json.dumps(self.body_template).format(input=input_path, output=output_path, **params))
                except Exception as e:
                    raise ValueError(f"Failed to format JSON body template: {e}")
            else:
                with open(input_path, "r") as f:
                    try:
                        json_body = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Input file is not valid JSON: {e}")

        elif self.payload_type.lower() == "binary":
            with open(input_path, "rb") as f:
                data_body = f.read()

        elif self.payload_type.lower() == "multipart":
            files_body = {"file": open(input_path, "rb")}

        else:
            raise ValueError(f"Unsupported payload_type '{self.payload_type}'.")

        try:
            logger.info(f"Sending {self.method} request to {endpoint_url} with timeout={self.timeout}s.")
            if self.method == "POST":
                response = requests.post(
                    endpoint_url,
                    headers=self.headers,
                    json=json_body,
                    data=data_body,
                    files=files_body,
                    timeout=self.timeout,
                )
            else:  # GET
                response = requests.get(
                    endpoint_url,
                    headers=self.headers,
                    timeout=self.timeout,
                )

            response.raise_for_status()

            # Save the response
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                output_file = os.path.join(output_path, "response.json")
                with open(output_file, "w") as f:
                    f.write(response.text)
            else:
                output_file = os.path.join(output_path, "response.bin")
                with open(output_file, "wb") as f:
                    f.write(response.content)

            return {"status": "success", "response": output_file}

        except Exception as exc:
            logger.exception("Error while executing REST API request.")
            raise RuntimeError(f"Error while executing REST API request: {exc}") from exc