.PHONY: create remove connect help setup

setup:
	ansible-galaxy collection install amazon.aws

create: setup
	ansible-playbook create_gpu_instance.yml

remove: setup
	ansible-playbook remove_gpu_instance.yml

connect:
	@echo "Getting instance name from vars/main.yml..."
	@INSTANCE_NAME=`grep "^instance_name:" vars/main.yml | cut -d ':' -f2 | tr -d ' '` ; \
	if [ -z "$$INSTANCE_NAME" ]; then \
		echo "Error: Instance name is empty in vars/main.yml"; \
		exit 1; \
	fi; \
	echo "Instance name: $$INSTANCE_NAME"; \
	echo "Finding running instance by name..."; \
	INSTANCE_ID=`aws ec2 describe-instances \
		--filters "Name=tag:Name,Values=$$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
		--query 'Reservations[0].Instances[0].InstanceId' \
		--output text \
		--profile virtual-cells-dev-poweruser` || { echo "Error: No running instance found with the specified name"; exit 1; }; \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "Error: No running instance found with the specified name"; \
		exit 1; \
	fi; \
	echo "Found instance ID: $$INSTANCE_ID"; \
	echo "Connecting to instance using AWS SSM (bash shell)..."; \
	aws ssm start-session \
		--target $$INSTANCE_ID \
		--document-name AWS-StartInteractiveCommand \
		--parameters command="bash -l" \
		--profile virtual-cells-dev-poweruser

help:
	@echo "Available targets:"
	@echo "  setup    - Install required Ansible collections"
	@echo "  create   - Create the GPU instance"
	@echo "  remove   - Remove the GPU instance and its security group"
	@echo "  connect  - Connect to the instance using AWS SSM"
	@echo "  help     - Show this help message" 