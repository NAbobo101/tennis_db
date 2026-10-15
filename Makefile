.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-slow check train-atp train-wta reproduce clean

UV ?= uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtual environment and install the git hooks
	$(UV) sync --locked
	$(UV) run pre-commit install

lint: ## Lint and check formatting
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Auto-fix lint findings and format the code
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck: ## Static type checking (mypy --strict)
	$(UV) run mypy

test: ## Fast test suite on synthetic data, with coverage
	$(UV) run pytest -m "not slow" --cov=tennis_predictor --cov-report=term-missing

test-slow: ## Integration tests on the full vendored dataset
	$(UV) run pytest -m slow

check: lint typecheck test ## Everything CI runs

train-atp: ## Train and evaluate the ATP model
	$(UV) run tennis train --config configs/atp.yaml

train-wta: ## Train and evaluate the WTA model
	$(UV) run tennis train --config configs/wta.yaml

reproduce: train-atp train-wta ## Rebuild every model and report from the raw data

clean: ## Remove caches, models and build outputs (keeps the committed reports)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find data/interim models -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
