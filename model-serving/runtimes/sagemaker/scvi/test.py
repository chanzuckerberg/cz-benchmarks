import boto3
import json
from utils import upload_to_s3
REGION = "us-west-2"

def test_endpoint(endpoint_name, payload):
    # Create a SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    inference_id, s3_uri = upload_to_s3(payload)
    print(f"Inference ID: {inference_id}")
    print(f"S3 URI: {s3_uri}")

    # Invoke the endpoint
    response = runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        InputLocation=s3_uri,
        InferenceId=inference_id,
        Accept="application/x-npy"
    )
    print(f"Inference ID: {response['InferenceId']}")
    print(f"Output Location: {response['OutputLocation']}")


def list_endpoints():
    client = boto3.client('sagemaker', region_name=REGION)
    response = client.list_endpoints()
    for endpoint in response['Endpoints']:
        print(endpoint['EndpointName'], endpoint['EndpointStatus'])


if __name__ == "__main__":
    endpoint_name = "scvi-endpoint"
    payload = json.dumps({
        "s3_input": "s3://generate-cross-species/datasets/test/example-small.h5ad",
        "organism": "homo_sapiens"
    })

    test_endpoint(endpoint_name, payload)
