.PHONY: create terminate connect help setup status stop start setup-schedule

setup:
	ansible-galaxy collection install amazon.aws community.aws

create: setup
	ansible-playbook create_gpu_instance.yml

terminate: setup
	ansible-playbook remove_gpu_instance.yml

connect:
	@echo "Getting instance name from vars/main.yml..."
	@USERNAME=`grep "^username:" vars/main.yml | cut -d ':' -f2 | tr -d ' '` ; \
	INSTANCE_NAME="$$USERNAME-gpu" ; \
	if [ -z "$$USERNAME" ]; then \
		echo "Error: Username is empty in vars/main.yml"; \
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

status:
	@echo "Getting instance name from vars/main.yml..."
	@USERNAME=`grep "^username:" vars/main.yml | cut -d ':' -f2 | tr -d ' '` ; \
	INSTANCE_NAME="$$USERNAME-gpu" ; \
	if [ -z "$$USERNAME" ]; then \
		echo "Error: Username is empty in vars/main.yml"; \
		exit 1; \
	fi; \
	echo "Instance name: $$INSTANCE_NAME"; \
	echo "Finding instance by name..."; \
	aws ec2 describe-instances \
		--filters "Name=tag:Name,Values=$$INSTANCE_NAME" \
		--query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,PrivateIpAddress]' \
		--output table \
		--profile virtual-cells-dev-poweruser

stop:
	@echo "Getting instance name from vars/main.yml..."
	@USERNAME=`grep "^username:" vars/main.yml | cut -d ':' -f2 | tr -d ' '` ; \
	INSTANCE_NAME="$$USERNAME-gpu" ; \
	if [ -z "$$USERNAME" ]; then \
		echo "Error: Username is empty in vars/main.yml"; \
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
	echo "Stopping instance..."; \
	aws ec2 stop-instances \
		--instance-ids $$INSTANCE_ID \
		--profile virtual-cells-dev-poweruser

start:
	@echo "Getting instance name from vars/main.yml..."
	@USERNAME=`grep "^username:" vars/main.yml | cut -d ':' -f2 | tr -d ' '` ; \
	INSTANCE_NAME="$$USERNAME-gpu" ; \
	if [ -z "$$USERNAME" ]; then \
		echo "Error: Username is empty in vars/main.yml"; \
		exit 1; \
	fi; \
	echo "Instance name: $$INSTANCE_NAME"; \
	echo "Finding stopped instance by name..."; \
	INSTANCE_ID=`aws ec2 describe-instances \
		--filters "Name=tag:Name,Values=$$INSTANCE_NAME" "Name=instance-state-name,Values=stopped" \
		--query 'Reservations[0].Instances[0].InstanceId' \
		--output text \
		--profile virtual-cells-dev-poweruser` || { echo "Error: No stopped instance found with the specified name"; exit 1; }; \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "Error: No stopped instance found with the specified name"; \
		exit 1; \
	fi; \
	echo "Found instance ID: $$INSTANCE_ID"; \
	echo "Starting instance..."; \
	aws ec2 start-instances \
		--instance-ids $$INSTANCE_ID \
		--profile virtual-cells-dev-poweruser

setup-schedule:
	ansible-playbook setup_schedule.yml

help:
	@echo "Available targets:"
	@echo "  setup      - Install required Ansible collections"
	@echo "  create     - Create the GPU instance"
	@echo "  terminate  - Terminate the GPU instance and its security group"
	@echo "  connect    - Connect to the instance using AWS SSM"
	@echo "  status     - Show the current status of the instance"
	@echo "  stop       - Stop the running instance"
	@echo "  start      - Start the stopped instance"
	@echo "  setup-schedule - Setup the EC2 instance schedule"
	@echo "  help       - Show this help message" 