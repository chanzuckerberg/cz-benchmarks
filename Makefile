# TODO: Default target

# Run all unit tests
.PHONY: test
test:
	uv run pytest

# Run all unit tests with coverage
.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage (minimum threshold: 74%)"
	uv run pytest --cov=czbenchmarks --cov-report=term-missing tests/ | tee coverage.txt
	@COVERAGE=$$(grep "TOTAL" coverage.txt | grep -o "[0-9]*%" | tr -d '%'); \
	if [ "$$COVERAGE" -lt 74 ]; then \
		echo "Test coverage ($$COVERAGE%) is below the required threshold of 74%"; \
		exit 1; \
	else \
		echo "Test coverage ($$COVERAGE%) meets the required threshold of 74%"; \
	fi

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
