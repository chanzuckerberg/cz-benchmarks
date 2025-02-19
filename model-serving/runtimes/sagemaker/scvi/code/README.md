# Sagemaker SCVI Model

This is a simple example of how to deploy a model using Sagemaker.

## Deploying the model
This will package the source code in `code/` and upload it to S3. It will also create a SageMaker model and deploy an endpoint.
If needed, it will also create a new IAM role which allows SageMaker to access the model artifacts and the data in S3.

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

```mermaid
graph TD
    A[Developer] -->|Deploy flow| B[make deploy]
    B --> C[Package Source Code in /code]
    C --> E[Upload to S3]
    E --> F[Check/Create IAM Role]
    F --> G[Create SageMaker Model]
    G --> H[Deploy SageMaker Endpoint]
    
    A -->|Test flow| I[make test]
    I --> K[Invoke SageMaker Endpoint]
    K --> L[SageMaker Processes Request]
    L --> M[Return Prediction Result]

    %% deploy.py
    subgraph deploy.py
        F
        G
        H
    end

    %% test.py
    subgraph test.py
        K
        L
        M
    end
```
