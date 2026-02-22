.DEFAULT_GOAL := help

PYTHON := python3.11
VENV := .venv

.PHONY: help
help: ## Show this help
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: ## Install package in development mode
	pip install -e ".[dev]"

.PHONY: format
format: ## Format source code
	ruff format src tests

.PHONY: lint
lint: ## Lint source code
	ruff check src tests

.PHONY: test
test: ## Run tests
	pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage
	pytest --cov=torchtomo --cov-report=term-missing

.PHONY: check
check: format lint test ## Run format, lint, and test

.PHONY: build
build: ## Build distribution packages
	$(PYTHON) -m build

.PHONY: ci
ci: check build ## Run all checks then build

.PHONY: publish-test
publish-test: build ## Upload to TestPyPI
	twine upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Upload to PyPI
	twine upload dist/*

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ __pycache__/ .pytest_cache/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: fresh
fresh: clean ## Create fresh venv and install from scratch
	rm -rf $(VENV)
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
