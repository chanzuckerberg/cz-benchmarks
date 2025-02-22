# SageMaker SCVI Model Serving

This directory contains the code for serving the SCVI model using SageMaker. Below you'll find info on core Sagemaker concepts and how the SCVI model is deployed.

## Overview
To deploy the SCVI model, run `make sagemaker-deploy`. This will:
* Package the `/code` directory into a .tar.gz file and upload it to S3.
* Create a SageMaker model package group (if it doesn't already exist) and register the model in the group.
* Create a SageMaker endpoint configuration
* Create a SageMaker endpoint (if it doesn't already exist)
* Deploy the model to the endpoint.

To test the model, run `make sagemaker-test`. This will:
* Upload the test data to S3.
* Invoke the endpoint with the S3 location of the test data as input.
* Receive as output the S3 location of the prediction results.
* Download the prediction results from S3 and print them (once the results are available).


## Sagemaker Concepts
Models in SageMaker are created by training or importing artifacts, registered in the Model Registry with metadata, packaged with inference code, then deployed to compute instances that serve predictions via REST API endpoints that we can monitor and update with zero downtime.

The following concepts are important to understand when working with SageMaker:

### Model Package

1. The trained model artifacts (weights, configuration) are packaged with an inference script
2. SageMaker creates a model package that combines the artifacts with container information 

### Model Package Group

1. You register your model in the SageMaker Model Registry, which acts as a catalog for models
2. Each model gets versioned and can have associated metadata for governance
3. Models can be grouped into "model groups" for organizational purposes
4. The registry tracks model lineage, approval status, and deployment history

### Endpoint Configuration

1. You define compute resources for the endpoint (instance type, count, etc.) and endpoint type (real-time, serverless, asynchronous, batch inference)
2. Configure auto-scaling policies if needed
3. Set up the variant structure if you want to perform A/B testing
4. Specify monitoring and logging preferences

### Endpoint Deployment

1. SageMaker provisions the specified EC2 instances
2. Container images are pulled and set up with your model artifacts
3. The inference code is configured to serve predictions
4. Health checks ensure the endpoint is ready to handle requests

### Runtime Operation

The model processes incoming prediction requests via a REST API endpoint that runs your inference code (code/inference.py).
Scaling occurs based on configuration (manual, auto-scaling, etc.)
CloudWatch metrics are available for monitoring the endpoint
