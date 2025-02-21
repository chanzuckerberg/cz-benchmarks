import boto3
import json
import numpy as np
from utils import upload_to_s3, download_s3_file, wait_for_s3_file
import time

REGION = "us-west-2"

def test_endpoint(endpoint_name, payload):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    inference_id, s3_uri = upload_to_s3(payload)
    print(f"Inference ID: {inference_id}")
    print(f"S3 URI: {s3_uri}")

    response = runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        InputLocation=s3_uri,
        InferenceId=inference_id,
        Accept="application/x-npy"
    )
    print(f"Inference ID: {response['InferenceId']}")
    print(f"Output Location: {response['OutputLocation']}")
    return response["OutputLocation"]


def list_endpoints():
    client = boto3.client('sagemaker', region_name=REGION)
    response = client.list_endpoints()
    for endpoint in response['Endpoints']:
        print(endpoint['EndpointName'], endpoint['EndpointStatus'])


if __name__ == "__main__":
    endpoint_name = "scvi-endpoint"
    payload = json.dumps({
        "s3_input": "s3://generate-cross-species/datasets/test/example-small.h5ad",
        "organism": "mus_musculus"
    })

    start_time = time.perf_counter()    
    output_location = test_endpoint(endpoint_name, payload)

    # Wait for the output file to be available
    try:
        wait_for_s3_file(output_location, timeout=3600, interval=1)  # Wait up to 1 hour, check every 10 seconds
    except TimeoutError as e:
        print(str(e))
        exit(1)

    download_s3_file(output_location, 'output.out')

    with open('output.out', 'rb') as f:
        buffer = f.read()

    npy_array = np.frombuffer(buffer, dtype=np.float32)
    print(npy_array)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Takes around 7 seconds for the example-small.h5ad dataset
    print(f"Execution Time: {elapsed_time:.2f} seconds")
