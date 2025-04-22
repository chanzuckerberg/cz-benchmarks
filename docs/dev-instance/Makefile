.PHONY: create remove connect help setup

setup:
	ansible-galaxy collection install amazon.aws

create: setup
	ansible-playbook create_gpu_instance.yml

remove: setup
	ansible-playbook remove_gpu_instance.yml

connect:
	@echo "Connecting to instance using AWS SSM..."
	@aws ssm start-session \
		--target $$(aws ec2 describe-instances \
			--filters "Name=tag:Name,Values=wcampos-gpu" \
			--query 'Reservations[0].Instances[0].InstanceId' \
			--output text \
			--profile virtual-cells-dev-poweruser) \
		--profile virtual-cells-dev-poweruser

help:
	@echo "Available targets:"
	@echo "  setup    - Install required Ansible collections"
	@echo "  create   - Create the GPU instance"
	@echo "  remove   - Remove the GPU instance and its security group"
	@echo "  connect  - Connect to the instance using AWS SSM"
	@echo "  help     - Show this help message" 