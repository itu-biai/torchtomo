.DEFAULT_GOAL := help

VENV := .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(PYTHON) -m pip
PYTEST := PYTHONPATH=src $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BUILD := $(PYTHON) -m build
TWINE := $(PYTHON) -m twine

.PHONY: help
help: ## Show this help
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: ## Install package in development mode
	$(PIP) install -e ".[dev]"

.PHONY: format
format: ## Format source code
	$(RUFF) format src tests benchmark

.PHONY: lint
lint: ## Lint source code
	$(RUFF) check src tests benchmark

.PHONY: test
test: ## Run tests
	$(PYTEST)

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(PYTEST) --cov=torchtomo --cov-report=term-missing

.PHONY: benchmark
benchmark: ## Run benchmark tests (requires benchmark extras)
	$(PYTEST) benchmark/

.PHONY: benchmark-speed
benchmark-speed: ## Run speed benchmarks (requires benchmark extras)
	PYTHONPATH=src $(PYTHON) benchmark/benchmark_speed.py

.PHONY: check
check: format lint test ## Run format, lint, and test

.PHONY: build
build: ## Build distribution packages
	$(BUILD)

.PHONY: ci
ci: check build ## Run all checks, then build

.PHONY: publish-test
publish-test: build ## Upload to TestPyPI
	$(TWINE) upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Upload to PyPI
	$(TWINE) upload dist/*

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ __pycache__/ .pytest_cache/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: fresh
fresh: clean ## Create fresh venv and install from scratch
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"
