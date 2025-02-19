import boto3
import json

REGION = "us-west-2"

def test_endpoint(endpoint_name, payload):
    # Create a SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )
    
    # Read and decode the response
    result = json.loads(response["Body"].read().decode())
    print(result)

if __name__ == "__main__":
    endpoint_name = "pytorch-inference-2025-02-19-22-01-07-113"
    payload = json.dumps({
        "s3_input": "s3://generate-cross-species/datasets/test/example-small.h5ad",
        "organism": "human"
    })

    test_endpoint(endpoint_name, payload)
