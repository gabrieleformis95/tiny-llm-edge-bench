PYTHON ?= .venv/bin/python

.PHONY: install-dev install lint type-check test bench-all mcu-bench report ui lock clean

install-dev:
	pip install -e ".[dev]"

install:
	pip install -e .

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

type-check:
	mypy src/

test:
	pytest tests/ --cov=src --cov-report=term-missing

bench-all:
	bash scripts/run_full_matrix.sh

mcu-bench:
	$(PYTHON) -m src.mcu.export
	make -C src/mcu/firmware
	$(PYTHON) -m src.mcu.renode_runner

report:
	$(PYTHON) scripts/generate_report.py

ui:
	streamlit run src/ui/streamlit_app.py

lock:
	pip freeze | sort > requirements-lock.txt
	@echo "Lockfile updated: requirements-lock.txt"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
