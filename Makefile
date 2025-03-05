# Default target
.PHONY: all
all: scvi uce scgpt scgenept

# Build the scvi image
.PHONY: scvi
scvi:
	docker build -t cz_benchmarks-scvi:latest -f docker/scvi/Dockerfile .

# Build the uce image
.PHONY: uce
uce:
	docker build -t cz_benchmarks-uce:latest -f docker/uce/Dockerfile .

# Build the scgpt image
.PHONY: scgpt
scgpt:
	docker build -t cz_benchmarks-scgpt:latest -f docker/scgpt/Dockerfile .

# Build the scgenept image
.PHONY: scgenept
scgenept:
	docker build -t cz_benchmarks-scgenept:latest -f docker/scgenept/Dockerfile .

# Build the geneformer image
.PHONY: geneformer
geneformer:
	docker build -t cz_benchmarks-geneformer:latest -f docker/geneformer/Dockerfile .

# Clean up images
.PHONY: clean
clean:
	docker rmi cz_benchmarks-scvi:latest || true
	docker rmi cz_benchmarks-uce:latest || true
	docker rmi cz_benchmarks-scgpt:latest || true
	docker rmi cz_benchmarks-scgenept:latest || true
  docker rmi cz_benchmarks-geneformer:latest || true

# Helper target to rebuild everything from scratch
.PHONY: rebuild
rebuild: clean all

# Check formatting with black
.PHONY: black-check
black-check:
	# Check if code conforms to Black's formatting style
	black --check .

# Fix formatting with black
.PHONY: black-fix
black-fix:
	# Automatically format code using Black
	black .

# Run flake8 to lint the code
.PHONY: flake8
flake8:
	# Lint the code using Flake8
	flake8

# Apply fixes for unused imports and variables
.PHONY: autoflake
autoflake:
	# Automatically remove unused imports and variables
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive .

# Run ruff to check the code
.PHONY: ruff-check
ruff-check:
	# Check code with Ruff
	ruff check .

# Run ruff with auto-fix
.PHONY: ruff-fix
ruff-fix:
	# Auto-fix code with Ruff
	ruff check . --fix

# Run mypy type checking
.PHONY: mypy-check
mypy-check:
	# Type check code with mypy
	mypy .

# Run all linters and checkers
.PHONY: lint
lint: flake8 black-check ruff-check #mypy-check

# Run all linters and fixers
.PHONY: lint-fix
lint-fix: autoflake black-fix ruff-fix

# Install tools explicitly
.PHONY: install-tools
install-tools:
	# Install all required linting and formatting tools
	pip install black flake8 autoflake
