SHELL := /bin/bash
PYTHON ?= python

.PHONY: help install install-dev format check test test-unit test-headless test-batch test-flexible
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  help: Show this help"
	@echo "  install: Install runtime dependencies"
	@echo "  install-dev: Install runtime, UI, and development dependencies"
	@echo "  format: Format the new brazing simulation and tests"
	@echo "  check: Run static checks without rewriting files"
	@echo "  test: Run the complete automated suite"
	@echo "  test-headless: Run the A-order headless smoke test"
	@echo "  test-batch: Run the three-layer A-batch headless smoke test"
	@echo "  test-flexible: Validate and execute all YAML-driven products"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[ui,dev]'

format:
	$(PYTHON) -m black brazing_line.py run_flexible_order.py brazing_sim tests
	$(PYTHON) -m ruff check --fix brazing_line.py run_flexible_order.py brazing_sim tests

check:
	$(PYTHON) -m ruff check brazing_line.py run_flexible_order.py brazing_sim tests
	$(PYTHON) -m black --check brazing_line.py run_flexible_order.py brazing_sim tests

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-headless:
	$(PYTHON) brazing_line.py --headless --order A --fast --max-sim-time 180

test-batch:
	$(PYTHON) brazing_line.py --headless --batch A --fast --max-sim-time 180

test-flexible:
	$(PYTHON) run_flexible_order.py --order config/orders/order_001.yaml --dry-run
	$(PYTHON) run_flexible_order.py --order config/orders/order_002.yaml --dry-run
	$(PYTHON) run_flexible_order.py --order config/orders/order_003.yaml --dry-run
