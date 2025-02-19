# Sagemaker SCVI Model

This is a simple example of how to deploy a model using Sagemaker.

## Deploying the model

```bash
make deploy
```

## Testing the model

```bash
make test
```

## Deployment & Testing Flow
High level summary: 
- Modify model code in `code/`
- Run `make deploy` to package and upload model code to S3 and deploy the model to SageMaker
- Run `make test` to test the model

Detailed flow:
```mermaid
graph TD
    A[Developer] -->|Run `make deploy`| B[Makefile: Deploy Target]
    B --> C[Package Source Code<br/>(`make package`)]
    C --> D[Upload to S3<br/>(`aws s3 cp`)]
    D --> E[Execute `deploy.py`]
    E --> F[Check/Create IAM Role<br/>(`create_sagemaker_role.py`)]
    F --> G[Create SageMaker Model<br/>(`PyTorchModel`)]
    G --> H[Deploy SageMaker Endpoint<br/>(`sagemaker.Session().deploy()`)]
    
    A -->|Run `make test`| I[Makefile: Test Target]
    I --> J[Execute `test.py`]
    J --> K[Invoke SageMaker Endpoint<br/>(`sagemaker-runtime.invoke_endpoint`)]
    K --> L[SageMaker Processes Request]
    L --> M[Return Prediction Result]
    
    %% AWS Services
    subgraph AWS
        D[S3 Bucket: omar-data]
        G[SageMaker]
        F[IAM Role]
    end
```
