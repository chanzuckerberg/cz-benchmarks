import json
import boto3

def create_sagemaker_execution_role(role_name="SageMakerExecutionRole"):
    """
    Creates an IAM role that SageMaker can assume, with permissions to
    read from S3 and use basic SageMaker features.

    Returns:
        The ARN of the created (or existing) IAM role.
    """
    iam = boto3.client("iam")

    # 1) Define the trust policy so SageMaker can assume the role
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    # Check if the role already exists
    try:
        existing_role = iam.get_role(RoleName=role_name)
        print(f"Role '{role_name}' already exists. ARN: {existing_role['Role']['Arn']}")
        return existing_role['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        pass  # Role does not exist, proceed to create it

    # 2) Create the role
    print(f"Creating IAM role '{role_name}'...")
    create_role_response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description="Role for SageMaker to access S3 and other AWS services"
    )
    role_arn = create_role_response["Role"]["Arn"]
    print(f"Created role: {role_arn}")

    # 3) Attach policies. 
    #    *At minimum*, attach AmazonS3FullAccess if you need S3 read and write permissions
    #    and a restricted SageMaker policy (like AmazonSageMakerFullAccess or a custom policy).
    #    For demonstration, we'll attach these AWS-managed policies:
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )

    print(f"Attached AmazonS3FullAccess and AmazonSageMakerFullAccess to {role_name}")

    # Give a little time for the role to fully propagate (sometimes needed)
    import time
    time.sleep(5)

    return role_arn
