SHELL := /bin/bash
PYTHON ?= python

.PHONY: help install install-dev format check test test-unit test-v1 test-headless test-batch test-cinematic test-flexible test-v2 experiment-v2 benchmark-help
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
	@echo "  test-v1: Run the standard-line and shared regression suites"
	@echo "  test-headless: Run the A-order headless smoke test"
	@echo "  test-batch: Run the three-layer A-batch headless smoke test"
	@echo "  test-cinematic: Compile and smoke-test the high-fidelity visual edition"
	@echo "  test-flexible: Validate and execute all YAML-driven products"
	@echo "  test-v2: Run the independent dual-install V2 regression suite"
	@echo "  experiment-v2: Run a 32x fixed/dynamic comparison"
	@echo "  benchmark-help: Show the reproducible V1/V2 benchmark options"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[ui,dev]'

format:
	$(PYTHON) -m black brazing_line.py brazing_line_v2.py brazing_line_cinematic.py run_flexible_order.py brazing_sim tests
	$(PYTHON) -m ruff check --fix brazing_line.py brazing_line_v2.py brazing_line_cinematic.py run_flexible_order.py brazing_sim tests

check:
	$(PYTHON) -m ruff check brazing_line.py brazing_line_v2.py brazing_line_cinematic.py run_flexible_order.py brazing_sim tests
	$(PYTHON) -m black --check brazing_line.py brazing_line_v2.py brazing_line_cinematic.py run_flexible_order.py brazing_sim tests

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest tests/shared tests/v1

test-v1:
	$(PYTHON) -m pytest tests/shared tests/v1

test-headless:
	$(PYTHON) brazing_line.py --headless --order A --fast --max-sim-time 180

test-batch:
	$(PYTHON) brazing_line.py --headless --batch A --fast --max-sim-time 180

test-cinematic:
	$(PYTHON) -m pytest tests/v1/test_cinematic_scene.py
	$(PYTHON) brazing_line_cinematic.py --headless --order A --fast --max-sim-time 300 --no-ui --no-terminal-commands

test-flexible:
	$(PYTHON) run_flexible_order.py --order config/orders/order_001.yaml --dry-run
	$(PYTHON) run_flexible_order.py --order config/orders/order_002.yaml --dry-run
	$(PYTHON) run_flexible_order.py --order config/orders/order_003.yaml --dry-run

test-v2:
	$(PYTHON) -m pytest tests/v2

experiment-v2:
	$(PYTHON) run_flexible_order.py --orders config/orders/batch_abc.yaml --compare --headless --speed 32

benchmark-help:
	$(PYTHON) benchmarks/compare_versions.py --help
