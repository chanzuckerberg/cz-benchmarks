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
    
    payload = json.dumps({
        "s3_input": "s3://generate-cross-species/datasets/test/example_small.h5ad",
        "organism": "mus_musculus",
    })

    start_time = time.perf_counter()
    
    try:
        result_bytes = test_local_endpoint(endpoint_url, payload)
        # Convert the binary response into a numpy array (assuming the endpoint returns an x-npy content)
        npy_array = np.frombuffer(result_bytes, dtype=np.float32)
        print("Inference output:", npy_array)
    except Exception as e:
        print("Error during inference:", e)
        exit(1)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")
