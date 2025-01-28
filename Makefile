.PHONY: all scvi uce scgpt clean check-tools black black-check black-fix flake8 autopep8 autoflake lint lint-fix install-tools rebuild

# Default target
all: scvi uce scgpt

# Build the scvi image
scvi:
	docker build -t czibench-scvi:latest -f docker/scvi/Dockerfile .

# Build the uce image
uce:
	docker build -t czibench-uce:latest -f docker/uce/Dockerfile .

# Build the scgpt image
scgpt:
	docker build -t czibench-scgpt:latest -f docker/scgpt/Dockerfile .

# Clean up images
clean:
	docker rmi czibench-scvi:latest || true
	docker rmi czibench-uce:latest || true
	docker rmi czibench-scgpt:latest || true

# Helper target to rebuild everything from scratch
rebuild: clean all

# Ensure all required tools are installed
check-tools:
	@command -v black >/dev/null 2>&1 || { echo >&2 "black not found, installing..."; pip install black; }
	@command -v flake8 >/dev/null 2>&1 || { echo >&2 "flake8 not found, installing..."; pip install flake8; }
	@command -v autopep8 >/dev/null 2>&1 || { echo >&2 "autopep8 not found, installing..."; pip install autopep8; }
	@command -v autoflake >/dev/null 2>&1 || { echo >&2 "autoflake not found, installing..."; pip install autoflake; }

# Check formatting with black
black-check: check-tools
	black --check .

# Fix formatting with black
black-fix: check-tools
	black .

# Run flake8 to lint the code
flake8: check-tools
	flake8 .

# Apply fixes for unused imports and variables
autoflake: check-tools
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive .

# Apply fixes for PEP8 issues
autopep8: check-tools
	autopep8 --in-place --recursive .

# Run all linters and checkers
lint: black-check flake8

# Run all linters and fixers
lint-fix: autoflake autopep8 black flake8

# Install tools explicitly
install-tools:
	pip install black flake8 autopep8 autoflake
