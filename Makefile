.PHONY: install format lint test clean

install:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

format:
	black omega_prm tests
	isort omega_prm tests

lint:
	flake8 omega_prm tests
	black --check omega_prm tests
	isort --check-only omega_prm tests
	mypy omega_prm

test:
	pytest tests/ -v --cov=omega_prm --cov-report=term-missing

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete