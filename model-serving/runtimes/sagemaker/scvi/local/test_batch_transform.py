# test_inference_batch.py

import os
import json
import logging
import time

from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel

ROLE = "OmarSageMakerRole"

# If your local model artifact is stored locally, use file:// prefix.
LOCAL_MODEL_ARTIFACT = "file://scvi_model_code.tar.gz"
MODEL_NAME = "scvi-local"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_local_batch_transform(input_file_uri, output_dir_uri, instance_type="local"):
    """
    Runs a local batch transform job. input_file_path should be a local text file
    containing JSON lines. Each line has "s3_input" + "organism".
    """
    # 1. Local session
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # 2. PyTorchModel referencing your inference.py that has input_fn, etc.
    pytorch_model = PyTorchModel(
        name=MODEL_NAME,
        model_data=LOCAL_MODEL_ARTIFACT,
        role=ROLE,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",  # see the example above
        source_dir="code/",
        sagemaker_session=sagemaker_session
    )

    # 3. Create a transformer
    transformer = pytorch_model.transformer(
        instance_count=1,
        instance_type=instance_type,  # "local" or "local_gpu"
        output_path=output_dir_uri,      # local path -> "file:///home/ssm-user/batch_transform_test/batch_output"
    )

    # 4. Transform job
    transformer.transform(
        data=input_file_uri,   # local path -> "file:///home/ssm-user/batch_transform_test/batch_input.jsonl"
        content_type="application/json",
        split_type="Line",      # Each line is one record for input_fn
        # accept="application/json", # TRY THIS LATER: "application/x-npy"
    )
    transformer.wait()
    logger.info("Local batch transform complete.")

def test_batch_transform():
    # We create a local "batch input" file with lines of JSON referencing S3 .h5ad
    # This is how we replicate your real-time approach but in a batch scenario.

    # 1. Create a local JSON lines file:
    input_filename = "/home/ssm-user/batch_transform_test/batch_input.jsonl"
    records = [
        {
            "s3_input": "s3://generate-cross-species/datasets/tsv2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Stomach_v2_curated.h5ad",
            "organism": "homo_sapiens"
        },
        # add more lines if you want multiple files or multiple invocations
        # {
        #     "s3_input": "s3://some-other-bucket/datasets/another_file.h5ad",
        #     "organism": "mus_musculus"
        # }
    ]

    with open(input_filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # 2. Set output path (can be a directory in your home directory)
    output_dir = "//home/ssm-user/batch_transform_test/batch_output"
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.perf_counter()
    # 3. Run local batch transform
    input_file_uri = f"file://{input_filename}"
    output_dir_uri = f"file://{output_dir}"
    run_local_batch_transform(
        input_file_uri=input_file_uri,
        output_dir_uri=output_dir_uri,
        instance_type="local_gpu"    # or "local" if you don't want GPU
    )

    # 4. Check the output
    # Typically the container will write output to
    # /home/ssm-user/batch_transform_test/batch_output
    # with .out suffix
    # For example, "batch_input.jsonl.out"
    out_file = "/home/ssm-user/batch_transform_test/batch_output/batch_input.jsonl.out"
    if not os.path.isfile(out_file):
        logger.error(f"No output found at {out_file}")
    else:
        logger.info(f"Reading batch transform output from {out_file}...")

        # Each line in the .out file corresponds to one record's output in JSON
        with open(out_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    result_json = json.loads(line)
                    # result_json might be a dict or list, depending on your output_fn
                    print(f"Record {idx} output:", result_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON on line {idx}: {e}")
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    test_batch_transform()