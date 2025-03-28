# Default target
.PHONY: all
all: scvi uce scgpt scgenept

# Build the scvi image
.PHONY: scvi
scvi:
	docker build -t czbenchmarks-scvi:latest -f docker/scvi/Dockerfile .

# Build the uce image
.PHONY: uce
uce:
	docker build -t czbenchmarks-uce:latest -f docker/uce/Dockerfile .

# Build the scgpt image
.PHONY: scgpt
scgpt:
	docker build -t czbenchmarks-scgpt:latest -f docker/scgpt/Dockerfile .

# Build the scgenept image
.PHONY: scgenept
scgenept:
	docker build -t czbenchmarks-scgenept:latest -f docker/scgenept/Dockerfile .

# Build the geneformer image
.PHONY: geneformer
geneformer:
	docker build -t czbenchmarks-geneformer:latest -f docker/geneformer/Dockerfile .

# Clean up images
.PHONY: clean
clean:
	docker rmi czbenchmarks-scvi:latest || true
	docker rmi czbenchmarks-uce:latest || true
	docker rmi czbenchmarks-scgpt:latest || true
	docker rmi czbenchmarks-scgenept:latest || true
  docker rmi czbenchmarks-geneformer:latest || true

# Helper target to rebuild everything from scratch
.PHONY: rebuild
rebuild: clean all

# Run all unit tests with coverage
.PHONY: test
test:
	uv run pytest --cov=czbenchmarks --cov-report=term-missing tests/

# Check formatting with ruff
.PHONY: ruff-fmt-check
ruff-fmt-check:
	uv run ruff format --check .

# Fix formatting with ruff
.PHONY: ruff-fmt
ruff-fmt:
	uv run ruff format .

# Run ruff to check the code
.PHONY: ruff-check
ruff-check:
	uv run ruff check .

# Run ruff with auto-fix
.PHONY: ruff-fix
ruff-fix:
	uv run ruff check . --fix

# Run mypy type checking
.PHONY: mypy-check
mypy-check:
	uv run mypy .

# Run all linters and checkers # TODO: enable mypy-check
.PHONY: lint
lint: ruff-check ruff-fmt-check #mypy-check
