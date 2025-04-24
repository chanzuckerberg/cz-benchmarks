# AWS GPU Instance Management

This repository contains Ansible playbooks to manage AWS GPU instances. It provides commands to create, connect to, and manage GPU instances with specific configurations.

## Prerequisites

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install required packages:
   ```bash
   brew install ansible awscli aws-oidc
   ```

3. Install AWS SSM Session Manager plugin:
   ```bash
   brew install --cask session-manager-plugin
   ```

## Local macOS Setup

1. Configure AWS credentials:
   ```bash
   aws-oidc configure --issuer-url https://czi-prod.okta.com --client-id aws-config --config-url https://aws-config-generation.prod.si.czi.technology
   ```

2. Install required Ansible collections:
   ```bash
   make setup
   ```

3. Review and modify variables in `vars/main.yml` as needed (see Configuration Variables section below)

## Configuration Variables

Before running the playbooks, review the variables in `vars/main.yml`. The following table shows which variables need to be changed and provides examples:

| Variable | Change Required | Example | Notes |
|----------|----------------|---------|-------|
| `username` | Yes | `<your-username>` | Your username for instance naming |
| `instance_name` | No | `{{ username }}-gpu-dev` | Automatically generated from username |
| `instance_type` | Optional | `g4dn.8xlarge` | Can be changed based on needs |
| `volume_size` | Optional | `200` | Storage size in GB |
| `volume_type` | Optional | `gp3` | Storage type |
| `aws_profile` | Optional | `virtual-cells-dev-poweruser` | Default profile |
| `aws_region` | No | `us-west-2` | Must remain as is |
| `ami_name` | No | `benchmarking-dev` | Must remain as is |
| `ami_owner_id` | No | `058264139299` | Must remain as is |
| `iam_instance_profile` | No | `atar-ssm` | Must remain as is |

Example configuration in `vars/main.yml`:
```yaml
# User Configuration
username: <your-username>        # MUST CHANGE to your username

# Instance Configuration
instance_name: "{{ username }}-gpu-dev"  # Automatically generated
instance_type: g4dn.8xlarge      # optional
volume_size: 200                 # optional
volume_type: gp3                 # optional

# AWS Configuration
aws_profile: virtual-cells-dev-poweruser  # optional
aws_region: us-west-2                     # keep as it is

# AMI Configuration
ami_name: benchmarking-dev                # keep as it is
ami_owner_id: 058264139299               # keep as it is

# IAM Configuration
iam_instance_profile: atar-ssm            # keep as it is

```

## Usage

The following commands are available:

1. Create the EC2 instance:
   ```bash
   make create
   ```

2. Connect to the instance:
   ```bash
   make connect
   ```

3. Check instance status:
   ```bash
   make status
   ```

4. Stop the instance:
   ```bash
   make stop
   ```

5. Start the instance:
   ```bash
   make start
   ```

6. Terminate the instance when done:
   ```bash
   make terminate
   ```

## Instance Configuration

The playbook creates an instance with the following specifications:
- Name: {username}-gpu-dev (automatically generated)
- AMI: benchmarking-dev (fixed)
- Instance type: g4dn.8xlarge (configurable)
- Storage: 200 GB (configurable)
- Region: us-west-2 (fixed)
- IAM Instance Profile: atar-ssm (fixed)
- Security Group: No inbound/outbound rules
- Tags:
  - Name: {username}-gpu-dev
  - role: gpu-dev-instance

## Troubleshooting

1. If you encounter permission issues:
   - Verify your AWS OIDC credentials are correctly configured
   - Check if your IAM user has the necessary permissions

2. If SSM connection fails:
   - Verify the instance is running
   - Check if the instance has the correct IAM role (atar-ssm)
   - Ensure the SSM agent is running on the instance

3. If Ansible collection installation fails:
   - Try installing with `--force` flag:
     ```bash
     ansible-galaxy collection install amazon.aws --force
     ```
