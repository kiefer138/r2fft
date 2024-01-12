PROJ_BASE=$(shell pwd)
PYTHONVER=python3.10
PYTHONENV=$(PROJ_BASE)/venv
VENVPYTHON=$(PYTHONENV)/bin/$(PYTHONVER)

.PHONY: build
build: develop
	@echo "Building wheel file..."
	@$(VENVPYTHON) setup.py bdist_wheel
	@echo "Done."

.PHONY: develop
develop: bootstrap
	@echo "Installing project in development mode..."
	@$(VENVPYTHON) -m pip install -e .
	@$(VENVPYTHON) -m pip install -r requirements-dev.txt
	@echo "Done."

.PHONY: bootstrap
bootstrap:
	@echo "Bootstrapping project..."
	@$(PYTHONVER) -m venv $(PYTHONENV)
	@$(VENVPYTHON) -m pip install --upgrade pip
	@$(VENVPYTHON) -m pip install -r requirements.txt
	@echo "Done."

.PHONY: test
test:
	@echo "Running type checks and tests..."
	@$(VENVPYTHON) -m mypy src/ test/
	@$(VENVPYTHON) -m pytest test/
	@echo "Done."

.PHONY: format
format:
	@echo "Running black formatter..."
	@$(VENVPYTHON) -m black src/ test/ setup.py
	@echo "Done."

.PHONY: clean
clean:
	@echo "Cleaning project..."
	@rm -rf $(PYTHONENV)
	@rm -rf **/*.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@rm -rf src/final_project/__pycache__
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf build
	@rm -rf dist
	@echo "Done."
