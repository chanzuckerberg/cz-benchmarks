import boto3
import json
import numpy as np
from czibench.datasets.types import DataType
import time
from czibench.runners.model_runner import ModelRunnerBase
import uuid
from urllib.parse import urlparse

REGION = "us-west-2"
S3_BUCKET = "omar-data"

class SageMakerRunner(ModelRunnerBase):
    """Handles model execution logic for a SageMaker model"""

    def _run_local(self, dataset):
        raise NotImplementedError("Local mode is not supported for SageMaker models.")

    def _run_remote(self, dataset):
        payload = json.dumps(
            {
                "s3_input": dataset.source_path,
                "organism": str(dataset.get_input(DataType.ORGANISM)),
            }
        )
        inference_id, s3_uri = upload_to_s3(payload)
        print(f"Inference ID: {inference_id}")
        print(f"Input Location: {s3_uri}")

        start_time = time.perf_counter()

        runtime = boto3.client("sagemaker-runtime", region_name=REGION)

        response = runtime.invoke_endpoint_async(
            EndpointName=self.model_endpoint,
            ContentType="application/json",
            InputLocation=s3_uri,
            InferenceId=inference_id,
            Accept="application/x-npy",
        )
        print(f"Inference ID: {response['InferenceId']}")
        print(f"Output Location: {response['OutputLocation']}")
        output_location = response["OutputLocation"]
        

        # Wait for the output file to be available
        try:
            wait_for_s3_file(
                output_location, timeout=3600, interval=1
            )  # Wait up to 1 hour, check every second
        except TimeoutError as e:
            print(str(e))
            exit(1)

        download_s3_file(output_location, "output.out")

        prediction = np.load("output.out")
        print(prediction)
        print(prediction.shape)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # Takes around 7 seconds for the example-small.h5ad dataset
        print(f"Execution Time: {elapsed_time:.2f} seconds")
        
        dataset.set_output(DataType.EMBEDDING, prediction)
        
        return dataset




# TODO: DRY! Below funcs copied from model-serving/runtimes/sagemaker/scvi/code/utils.py

def upload_to_s3(payload, prefix="scvi-async-input/"):
    s3_client = boto3.client("s3", region_name=REGION)
    inference_id = str(uuid.uuid4())
    s3_key = f"{prefix}{inference_id}.json"
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=payload)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    return inference_id, s3_uri

def download_s3_file(s3_uri, local_filename):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", region_name=REGION)
    s3.download_file(bucket, key, local_filename)


def wait_for_s3_file(s3_uri, timeout=3600, interval=300):
    """
    Waits for a file to become available in S3.

    :param s3_uri: The S3 URI of the file to wait for.
    :param timeout: Maximum time to wait in seconds (default: 1 hour).
    :param interval: Time between checks in seconds (default: 5 minutes).
    :raises TimeoutError: If the file is not available within the timeout period.
    """
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3", region_name=REGION)

    start_time = time.time()
    while True:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"File {s3_uri} is now available.")
            break
        except boto3.exceptions.botocore.client.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Timeout: {s3_uri} not available after {timeout} seconds."
                    )
                print(
                    f"File {s3_uri} not found. Waiting for {interval} seconds before retrying..."
                )
                time.sleep(interval)
            else:
                raise
