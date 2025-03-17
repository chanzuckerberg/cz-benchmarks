import requests
import json
import numpy as np
import time

def test_local_endpoint(endpoint_url, payload):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/x-npy"
    }
    print(f"Sending payload to local endpoint at: {endpoint_url}")
    response = requests.post(endpoint_url, data=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}:\n{response.text}")
    
    print("Received response from local endpoint")
    return response.content

if __name__ == "__main__":
    # The local endpoint URL is typically hosted on port 8080 in local mode.
    endpoint_url = "http://localhost:8080/invocations"
    """
    Some files that have been confirmed to work:
    
    1.9 GB: s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Mammary_v2_curated.h5ad
    2.5 GB: s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Stomach_v2_curated.h5ad
    5.0 GB: s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Lymph_Node_v2_curated.h5ad
    6.0 GB: s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Fat_v2_curated.h5ad
    """
    payload = json.dumps({
        "s3_input": "s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Lymph_Node_v2_curated.h5ad",
        "organism": "homo_sapiens",
    })

    start_time = time.perf_counter()
    
    try:
        result_bytes = test_local_endpoint(endpoint_url, payload)
        # Convert the binary response into a numpy array (assuming the endpoint returns an x-npy content)
        npy_array = np.frombuffer(result_bytes, dtype=np.float32)
        print("Inference output:", npy_array)
        print(f"Inference output numpy array shape: {npy_array.shape}")
    except Exception as e:
        print("Error during inference:", e)
        exit(1)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")
