import boto3
import json

REGION = "us-west-2"
ENDPOINT_NAME = "pytorch-inference-2025-02-19-22-01-07-113"

def test_endpoint(s3_input, organism):
    # Create a SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    
    # Prepare the payload as JSON
    payload = json.dumps({
        "s3_input": s3_input,
        "organism": organism
    })
    
    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    
    # Read and decode the response
    result = json.loads(response["Body"].read().decode())
    print(result)

if __name__ == "__main__":
    s3_input = "s3://generate-cross-species/datasets/test/example-small.h5ad"
    organism = "human"
    test_endpoint(s3_input, organism)
