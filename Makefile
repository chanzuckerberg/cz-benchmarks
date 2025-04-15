# Default target
.PHONY: all
all: scvi uce scgpt scgenept geneformer transcriptformer

# Build the scvi image
.PHONY: scvi
scvi:
	docker build -t cz-benchmarks-models:scvi -f docker/scvi/Dockerfile .

# Build the uce image
.PHONY: uce
uce:
	docker build -t cz-benchmarks-models:uce -f docker/uce/Dockerfile .

# Build the scgpt image
.PHONY: scgpt
scgpt:
	docker build -t cz-benchmarks-models:scgpt -f docker/scgpt/Dockerfile .

# Build the scgenept image
.PHONY: scgenept
scgenept:
	docker build -t cz-benchmarks-models:scgenept -f docker/scgenept/Dockerfile .

# Build the geneformer image
.PHONY: geneformer
geneformer:
	docker build -t cz-benchmarks-models:geneformer -f docker/geneformer/Dockerfile .

# Build the transcriptformer image
.PHONY: transcriptformer
transcriptformer:
	docker build \
		--secret id=github_username,src=${GITHUB_USERNAME} \
		--secret id=github_token,src=${GITHUB_TOKEN} \
		-t cz-benchmarks-models:transcriptformer \
		-f docker/transcriptformer/Dockerfile .

# Clean up images
.PHONY: clean
clean:
	docker rmi cz-benchmarks-models:scvi || true
	docker rmi cz-benchmarks-models:uce || true
	docker rmi cz-benchmarks-models:scgpt || true
	docker rmi cz-benchmarks-models:scgenept || true
	docker rmi cz-benchmarks-models:geneformer || true
	docker rmi cz-benchmarks-models:transcriptformer || true
# Helper target to rebuild everything from scratch
.PHONY: rebuild
rebuild: clean all

# Run all unit tests
.PHONY: test
test:
	uv run pytest

# Run all unit tests with coverage
.PHONY: test-coverage
test-coverage:
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

.PHONY: lint-fix
lint-fix: ruff-fix ruff-fmt
